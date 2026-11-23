#!/usr/bin/env python3
"""Audit a chat corpus without loading the model.

The report answers the questions that a training loss cannot: how many examples
exceed the context or target budget, how many calls a row teaches, whether a
tool is actually executable, whether a prompt leaked in from an evaluation
suite, and whether the train/validation split overlaps.

    assistant/.venv/bin/python brain-lab/audit_corpus.py \
        --corpus brain-lab/data/chat-v16 --out runtime-data/eval/corpus-audit.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from claire_brain.prompt import render_prompt
from claireai.tools import default_registry

from fetch_corpora import FetchError, _find_tokenizer_manifest, _load_tokenizer, _parse_hermes_calls
from merge_corpora import held_out_prompts

EXIT_OK = 0
EXIT_VIOLATIONS = 1
EXIT_CONFIG = 2


def _corpus_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        return []
    preferred = [path / "train.jsonl", path / "validation.jsonl"]
    if any(item.is_file() for item in preferred):
        return [item for item in preferred if item.is_file()]
    return sorted(
        item
        for item in path.glob("*.jsonl")
        if item.is_file() and not item.name.endswith(".manifest.json")
    )


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    malformed = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            malformed += 1
            continue
        if not isinstance(record, dict) or not isinstance(record.get("messages"), list):
            malformed += 1
            continue
        records.append(record)
    return records, malformed


def _stats(values: Sequence[int]) -> dict[str, int]:
    if not values:
        return {"min": 0, "median": 0, "p90": 0, "max": 0}
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "median": ordered[len(ordered) // 2],
        "p90": ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))],
        "max": ordered[-1],
    }


def _call_names(content: str) -> tuple[list[str], bool]:
    if "<tool_call>" not in content:
        return [], False
    calls = _parse_hermes_calls(content)
    if calls is None:
        return [], True
    return [str(call["name"]) for call in calls], False


def _pair(record: dict[str, Any]) -> tuple[str, str] | None:
    messages = record.get("messages") or []
    user = next(
        (
            str(message.get("content") or "").strip()
            for message in messages
            if isinstance(message, dict) and message.get("role") == "user"
        ),
        "",
    )
    assistant = next(
        (
            str(message.get("content") or "").strip()
            for message in reversed(messages)
            if isinstance(message, dict) and message.get("role") == "assistant"
        ),
        "",
    )
    return (user, assistant) if user and assistant else None


def audit_records(
    records: Sequence[dict[str, Any]],
    tokenizer: Any,
    *,
    context: int = 512,
    target_cap: int = 192,
    held_out: Sequence[str] = (),
    local_tools: Sequence[str] = (),
) -> tuple[dict[str, Any], set[tuple[str, str]]]:
    """Return one report and the exact pairs it saw, for split-overlap checks."""
    held_out_set = set(held_out)
    local_set = set(local_tools)
    totals = Counter()
    prompt_tokens: list[int] = []
    target_tokens: list[int] = []
    total_tokens: list[int] = []
    calls = Counter()
    unknown_tools: Counter[str] = Counter()
    unknown_available: Counter[str] = Counter()
    duplicates = Counter()
    pairs: set[tuple[str, str]] = set()

    for record in records:
        totals["records"] += 1
        messages = record.get("messages") or []
        system = str(record.get("system") or "")
        if "TOOLS:" not in system:
            totals["missing_tool_block"] += 1

        user_messages = [
            str(message.get("content") or "").strip()
            for message in messages
            if isinstance(message, dict) and message.get("role") == "user"
        ]
        if any(prompt in held_out_set for prompt in user_messages):
            totals["held_out_overlap"] += 1

        available = record.get("available_tools")
        if isinstance(available, list):
            for name in available:
                if str(name) not in local_set:
                    unknown_available[str(name)] += 1

        pair = _pair(record)
        if pair is not None:
            duplicates[pair] += 1
            pairs.add(pair)

        assistant_index = next(
            (
                index
                for index in range(len(messages) - 1, -1, -1)
                if isinstance(messages[index], dict) and messages[index].get("role") == "assistant"
            ),
            None,
        )
        if assistant_index is None:
            totals["missing_assistant"] += 1
            continue
        target = str(messages[assistant_index].get("content") or "").strip()
        if not target:
            totals["missing_target"] += 1
            continue

        prompt = render_prompt(system, messages[:assistant_index])
        prompt_count = len(tokenizer.encode(prompt, append_eos=False))
        target_count = len(tokenizer.encode(target, append_eos=True))
        prompt_tokens.append(prompt_count)
        target_tokens.append(target_count)
        total_tokens.append(prompt_count + target_count)
        if prompt_count + target_count > context:
            totals["over_context"] += 1
        if target_count > target_cap:
            totals["over_target"] += 1

        names, malformed = _call_names(target)
        if malformed:
            totals["malformed_tool_calls"] += 1
        if len(names) > 2:
            totals["more_than_two_calls"] += 1
        calls[str(min(len(names), 2)) if len(names) <= 2 else "3+"] += 1
        for name in names:
            if name not in local_set:
                unknown_tools[name] += 1

    totals["duplicate_records"] = sum(count - 1 for count in duplicates.values())
    totals["unknown_tool_names"] = sum(unknown_tools.values())
    totals["unknown_available_names"] = sum(unknown_available.values())
    report = {
        "records": totals["records"],
        "calls": dict(sorted(calls.items())),
        "prompt_tokens": _stats(prompt_tokens),
        "target_tokens": _stats(target_tokens),
        "total_tokens": _stats(total_tokens),
        "over_context": totals["over_context"],
        "over_target": totals["over_target"],
        "more_than_two_calls": totals["more_than_two_calls"],
        "malformed_tool_calls": totals["malformed_tool_calls"],
        "unknown_tools": dict(unknown_tools.most_common()),
        "unknown_available_tools": dict(unknown_available.most_common()),
        "duplicate_records": totals["duplicate_records"],
        "held_out_overlap": totals["held_out_overlap"],
        "missing_tool_block": totals["missing_tool_block"],
        "missing_assistant": totals["missing_assistant"],
        "missing_target": totals["missing_target"],
    }
    return report, pairs


def _render(report: dict[str, Any], label: str) -> str:
    total = report["total_tokens"]
    target = report["target_tokens"]
    return (
        f"{label}: {report['records']} records; "
        f"total tokens {total['min']}/{total['median']}/{total['p90']}/{total['max']} "
        f"min/median/p90/max; target {target['min']}/{target['median']}/{target['p90']}/{target['max']}; "
        f"calls {report['calls'] or '{}'}; over context {report['over_context']}, "
        f"over target {report['over_target']}, duplicates {report['duplicate_records']}, "
        f"held-out {report['held_out_overlap']}, unknown tools {report['unknown_tools'] or '{}'}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path(__file__).resolve().parent / "data" / "chat-v16")
    parser.add_argument("--tokenizer-manifest", type=Path, default=None)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--target-tokens", type=int, default=192)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--strict", action="store_true", help="exit 1 on a corpus violation")
    args = parser.parse_args(argv)
    if args.context < 1 or args.target_tokens < 1:
        parser.error("--context and --target-tokens must be positive")

    files = _corpus_files(args.corpus.expanduser())
    if not files:
        print(f"ERROR: no JSONL corpus files under {args.corpus}", file=sys.stderr)
        return EXIT_CONFIG
    manifest = _find_tokenizer_manifest(args.tokenizer_manifest)
    if manifest is None:
        print("ERROR: no tokenizer manifest found; pass --tokenizer-manifest", file=sys.stderr)
        return EXIT_CONFIG
    try:
        tokenizer = _load_tokenizer(manifest)
    except FetchError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_CONFIG

    registry = default_registry(browser_enabled=True)
    local_tools = registry.names()
    held_out = sorted(held_out_prompts())
    reports: dict[str, Any] = {}
    pairs_by_file: dict[str, set[tuple[str, str]]] = {}
    malformed_lines = 0
    for path in files:
        try:
            records, malformed = _read_jsonl(path)
        except OSError as exc:
            print(f"ERROR: cannot read {path}: {exc}", file=sys.stderr)
            return EXIT_CONFIG
        malformed_lines += malformed
        report, pairs = audit_records(
            records,
            tokenizer,
            context=args.context,
            target_cap=args.target_tokens,
            held_out=held_out,
            local_tools=local_tools,
        )
        report["malformed_lines"] = malformed
        reports[path.name] = report
        pairs_by_file[path.name] = pairs
        print(_render(report, path.name))

    split_overlap = 0
    if "train.jsonl" in pairs_by_file and "validation.jsonl" in pairs_by_file:
        split_overlap = len(pairs_by_file["train.jsonl"] & pairs_by_file["validation.jsonl"])
    print(f"train/validation exact pair overlap: {split_overlap}")
    print(f"tokenizer: {manifest}")

    violations: list[str] = []
    for filename, report in reports.items():
        violations.extend(
            [f"{filename}: malformed line"] * report["malformed_lines"]
            + [f"{filename}: over context"] * report["over_context"]
            + [f"{filename}: over target"] * report["over_target"]
            + [f"{filename}: malformed tool call"] * report["malformed_tool_calls"]
            + [f"{filename}: more than two calls"] * report["more_than_two_calls"]
            + [f"{filename}: held-out overlap"] * report["held_out_overlap"]
            + [f"{filename}: duplicate record"] * report["duplicate_records"]
        )
        violations.extend(
            f"{filename}: unknown tool {name}" for name in report["unknown_tools"]
        )
        violations.extend(
            f"{filename}: unknown available tool {name}"
            for name in report["unknown_available_tools"]
        )
    if split_overlap:
        violations.append("train/validation overlap")
    if violations:
        print(f"violations: {len(violations)}")
    payload = {
        "corpus": str(args.corpus),
        "tokenizer": str(manifest),
        "context_tokens": args.context,
        "target_tokens": args.target_tokens,
        "files": reports,
        "train_validation_overlap": split_overlap,
        "violations": violations,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"report saved: {args.out}")
    return EXIT_VIOLATIONS if args.strict and violations else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
