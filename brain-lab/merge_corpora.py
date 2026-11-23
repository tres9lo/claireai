#!/usr/bin/env python3
"""Merge the curated corpus with downloaded public instruction samples.

The base corpus (chat-v14) carries the assistant's own formats and the licensed
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
DEFAULT_EXTERNAL_SHARE = 0.35


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
    for name in (
        "model_eval_cases.json",
        "conversation_cases.json",
        "tool_call_cases.json",
        "tool_call_cases_extended.json",
    ):
        path = ROOT.parent / "scripts" / name
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
            continue
        for case in payload["cases"]:
            if not isinstance(case, dict):
                continue
            for key in ("prompt", "user"):
                if case.get(key):
                    prompts.add(str(case[key]).strip())
            turns = case.get("turns")
            if not isinstance(turns, list):
                continue
            for turn in turns:
                if isinstance(turn, dict) and turn.get("user"):
                    prompts.add(str(turn["user"]).strip())
    return prompts


def _key(record: dict[str, Any]) -> tuple[str, str]:
    messages = record.get("messages") or []
    user = str(messages[0].get("content") or "").strip() if messages else ""
    assistant = str(messages[-1].get("content") or "").strip() if messages else ""
    return user, assistant


def dedupe(
    records: Iterable[dict[str, Any]],
    *,
    skip_prompts: set[str] | frozenset[str] = frozenset(),
    exclude: set[tuple[str, str]] | frozenset[tuple[str, str]] = frozenset(),
) -> tuple[list[dict[str, Any]], int]:
    """Drop malformed, held-out and exact duplicate pairs without merging stories.

    The same user prompt with different answers can be legitimate ("tell me a
    short story"). An exact duplicate pair is not: it inflated the old corpus by
    thousands of rows and made a first-batch loss of 0.0001 look plausible.
    """
    seen: set[tuple[str, str]] = set()
    kept: list[dict[str, Any]] = []
    dropped = 0
    for record in records:
        messages = record.get("messages") or []
        if len(messages) < 2:
            dropped += 1
            continue
        user, assistant = _key(record)
        if not user or not assistant or user in skip_prompts:
            dropped += 1
            continue
        pair = (user, assistant)
        if pair in seen or pair in exclude:
            dropped += 1
            continue
        seen.add(pair)
        kept.append(record)
    return kept, dropped


def cap_external(
    records: Iterable[dict[str, Any]], limit: int, *, seed: int = SEED
) -> list[dict[str, Any]]:
    """Keep external data a minority while preserving every source.

    ``--limit`` is per source, so five sources can still produce tens of
    thousands of rows and drown the authored conversation data.  This caps the
    total and samples proportionally from each source instead of letting raw
    web prose determine most of the training loss.
    """
    material = list(records)
    if limit < 0:
        raise ValueError("external cap must not be negative")
    if len(material) <= limit:
        return material
    if limit == 0:
        return []

    groups: dict[str, list[dict[str, Any]]] = {}
    for record in material:
        groups.setdefault(str(record.get("source") or "external"), []).append(record)
    if limit < len(groups):
        # More sources than rows: take one row from the largest sources first.
        names = sorted(groups, key=lambda name: (-len(groups[name]), name))[:limit]
        return [groups[name][0] for name in names]

    total = len(material)
    quotas = {name: max(1, int(limit * len(items) / total)) for name, items in groups.items()}
    while sum(quotas.values()) > limit:
        name = max((item for item in quotas if quotas[item] > 1), key=lambda item: quotas[item])
        quotas[name] -= 1
    while sum(quotas.values()) < limit:
        name = max(
            groups,
            key=lambda item: (len(groups[item]) - quotas[item], len(groups[item]), item),
        )
        if quotas[name] >= len(groups[name]):
            break
        quotas[name] += 1

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for name in sorted(groups):
        count = min(quotas[name], len(groups[name]))
        selected.extend(rng.sample(groups[name], count))
    return selected


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
    parser.add_argument("--base", type=Path, default=ROOT / "data" / "chat-v14")
    parser.add_argument("--external", type=Path, default=ROOT / "data" / "external")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "chat-v16")
    parser.add_argument("--limit", type=int, default=4000, help="maximum records per external source")
    parser.add_argument(
        "--external-share",
        type=float,
        default=DEFAULT_EXTERNAL_SHARE,
        help="maximum fraction of merged training records from external sources (default: 0.35)",
    )
    parser.add_argument("--base-limit", type=int, default=0, help="optional cap on base training records (0 = all)")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if not 0.0 <= args.external_share < 1.0:
        parser.error("--external-share must be at least 0 and below 1")

    base_train = read_jsonl(args.base / "train.jsonl")
    base_validation = read_jsonl(args.base / "validation.jsonl")
    if not base_train or not base_validation:
        parser.error(f"{args.base} needs train.jsonl and validation.jsonl")
    if args.base_limit and len(base_train) > args.base_limit:
        rng0 = random.Random(args.seed)
        base_train = rng0.sample(base_train, args.base_limit)
    # The authored builder emits some exact duplicate pairs. Deduplicate before
    # sizing the external quota; otherwise duplicates inflate the base and let
    # raw web text take more than the configured share of the final corpus.
    base_train_unique, base_duplicates = dedupe(base_train)
    external, manifests = load_external(args.external, max(1, args.limit))
    if not external:
        parser.error(f"no licensed external JSONL with sidecars under {args.external}; run fetch_corpora.py")

    held_out = held_out_prompts()
    rng = random.Random(args.seed)
    clean_external, external_dropped = dedupe(external, skip_prompts=held_out)
    # A per-source limit is not a total limit.  With the full profile, five
    # sources produced 17,459 external rows against 13,440 authored rows, and
    # validation loss was dominated by web prose rather than assistant replies.
    max_external = int(
        len(base_train_unique) * args.external_share / (1.0 - args.external_share)
    )
    selected_external = cap_external(clean_external, max_external, seed=args.seed)
    selected_by_source: dict[str, int] = {}
    for record in selected_external:
        name = str(record.get("source") or "external")
        selected_by_source[name] = selected_by_source.get(name, 0) + 1

    rng.shuffle(selected_external)
    split = max(1, len(selected_external) // 10)
    ext_validation, _ = dedupe(selected_external[:split])
    ext_train, _ = dedupe(selected_external[split:])

    validation, validation_dropped = dedupe(base_validation + ext_validation)
    # Never train on a pair that also appears in validation: that is evaluation
    # leakage, and it makes the validation loss look better than the model is.
    validation_pairs = {_key(record) for record in validation}
    train, train_dropped = dedupe(
        base_train_unique + ext_train, skip_prompts=held_out, exclude=validation_pairs
    )
    rng.shuffle(train)
    rng.shuffle(validation)
    write_jsonl(args.output / "train.jsonl", train)
    write_jsonl(args.output / "validation.jsonl", validation)

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = {
        "dataset": "claireai-chat-v16",
        "built_from": str(args.base),
        "external": manifests,
        "records_by_split": {"train": len(train), "validation": len(validation)},
        "external_records": len(clean_external),
        "external_records_selected": len(selected_external),
        "external_records_by_source": dict(sorted(selected_by_source.items())),
        "external_share": args.external_share,
        "base_train_records": len(base_train),
        "base_train_unique_records": len(base_train_unique),
        "base_duplicate_records": base_duplicates,
        "held_out_prompts_removed": len(held_out),
        "duplicate_or_overlap_removed": {
            "external": external_dropped,
            "train": train_dropped,
            "validation": validation_dropped,
        },
        "seed": args.seed,
        "licenses": {
            name: {"license": item.get("license"), "url": item.get("url"), "sha256": item.get("sha256")}
            for name, item in manifests.items()
        },
        "notes": [
            "External records are bounded samples, not a full corpus.",
            "External records are capped to a minority of training data so authored conversation and tool records are not drowned out.",
            "Evaluation prompts are removed by exact match; run the full gate after training.",
            "The gate, not this manifest, decides whether a checkpoint may be installed.",
        ],
        "files": {
            "train": {"sha256": digest(args.output / "train.jsonl"), "records": len(train)},
            "validation": {"sha256": digest(args.output / "validation.jsonl"), "records": len(validation)},
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(train)} train / {len(validation)} validation records to {args.output}")
    print(
        f"external: {len(selected_external)}/{len(clean_external)} kept "
        f"({args.external_share:.0%} max) from {', '.join(manifests)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
