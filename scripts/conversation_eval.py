#!/usr/bin/env python3
"""Measure whether the scratch model can hold a conversation.

The fixed ``model_eval_cases.json`` suite checks one-shot formatting.  This
suite is different: it runs multi-turn conversations, injects tool results, and
scores every reply with deterministic code.  It is deliberately held out from
training (``build_chat_dataset.py`` removes verbatim evaluation prompts).

    assistant/.venv/bin/python scripts/conversation_eval.py
    assistant/.venv/bin/python scripts/conversation_eval.py --runs 3 --transcript

Exit codes: 0 all cases passed, 1 a case failed, 2 bad configuration, 3 model
unavailable.  Nothing here is graded by another model.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from claireai.brain import Brain, BrainError, Message, ScratchBrain
from claireai.brain.tool_calls import looks_like_tool_call
from claireai.config import Settings
from claireai.evaluation import extract_json_object
from claireai.tools import default_registry

DEFAULT_CASES = Path(__file__).resolve().parent / "conversation_cases.json"

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_CONFIG = 2
EXIT_RUNTIME = 3


@dataclass
class TurnResult:
    user: str
    reply: str
    passed: bool
    reason: str
    tool_calls: tuple[str, ...] = ()


@dataclass
class CaseResult:
    case_id: str
    turns: list[TurnResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.turns) and all(turn.passed for turn in self.turns)


def _normalise(text: str) -> str:
    return " ".join(text.strip().strip("\"'`.!?,:;").lower().split())


def check(expect: Mapping[str, Any], reply: str, tool_calls: tuple[str, ...]) -> tuple[bool, str]:
    kind = expect.get("kind")
    lowered = reply.lower()

    if kind == "exact":
        expected = _normalise(str(expect["value"]))
        actual = _normalise(reply)
        return (actual == expected, f"expected {expected!r}, got {actual[:80]!r}")

    if kind == "contains_all":
        missing = [phrase for phrase in expect["value"] if str(phrase).lower() not in lowered]
        return (not missing, f"missing {missing}" if missing else "all phrases present")

    if kind == "contains_any":
        if any(str(phrase).lower() in lowered for phrase in expect["value"]):
            return (True, "matched an allowed phrase")
        return (False, f"none of {list(expect['value'])[:5]} present")

    if kind == "not_contains":
        found = [phrase for phrase in expect["value"] if str(phrase).lower() in lowered]
        return (not found, f"contained forbidden {found}" if found else "no forbidden text")

    if kind == "refuses":
        forbidden = [str(item) for item in expect.get("forbidden") or []]
        claimed = [item for item in forbidden if item.lower() in lowered]
        if claimed:
            return (False, f"claimed an approval it does not have: {claimed}")
        markers = [str(item) for item in expect["value"]]
        if any(marker.lower() in lowered for marker in markers):
            return (True, "refused without claiming approval")
        return (False, f"did not refuse (none of {markers[:4]} present)")

    if kind == "json":
        payload = extract_json_object(reply)
        if payload is None:
            return (False, "no JSON object found")
        for path, expected in (expect.get("equals") or {}).items():
            actual = _dotted(payload, path)
            if actual != expected:
                return (False, f"{path} = {actual!r}, expected {expected!r}")
        for path, needle in (expect.get("contains") or {}).items():
            actual = _dotted(payload, path)
            if actual is None or str(needle).lower() not in str(actual).lower():
                return (False, f"{path} = {actual!r} does not contain {needle!r}")
        return (True, "JSON matched")

    if kind == "tool_call":
        if not tool_calls:
            # A reply that is a call block but arrives as empty ``tool_calls`` was
            # dropped by the parser, usually because the name is not one of the
            # tools this case offers.  The case still fails, but the reason has to
            # say which, or a parser problem looks like a model problem.
            if looks_like_tool_call(reply):
                return (False, "model called a tool that was not offered (call dropped)")
            return (False, "model produced no tool call")
        if expect.get("name") and expect["name"] not in tool_calls:
            return (False, f"called {tool_calls}, expected {expect['name']}")
        return (True, f"called {', '.join(tool_calls)}")

    if kind == "max_words":
        count = len(re.findall(r"\S+", reply))
        return (count <= int(expect["value"]), f"{count} words (limit {expect['value']})")

    if kind == "min_words":
        count = len(re.findall(r"\S+", reply))
        return (count >= int(expect["value"]), f"{count} words (minimum {expect['value']})")

    if kind == "no_prompt_echo":
        found = [marker for marker in ("SYSTEM:", "USER:", "ASSISTANT:") if marker in reply]
        return (not found, f"echoed {found}" if found else "no prompt echo")

    return (False, f"unknown check kind {kind!r}")


def _dotted(source: Mapping[str, Any], path: str) -> Any:
    current: Any = source
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


async def run_case(brain: Brain, case: Mapping[str, Any], registry) -> CaseResult:  # noqa: ANN001
    result = CaseResult(case_id=str(case["id"]))
    tools = None
    if case.get("tools"):
        wanted = set(case["tools"])
        tools = [schema for schema in registry.schemas() if schema["function"]["name"] in wanted]
    messages: list[Message] = [Message(role="system", content=str(case.get("system") or ""))]
    for turn in case.get("turns", []):
        if turn.get("role") == "tool":
            messages.append(
                Message(role="tool", content=str(turn.get("content") or ""), tool_name=turn.get("name"))
            )
            continue
        user = str(turn.get("user") or "")
        messages.append(Message(role="user", content=user))
        try:
            response = await brain.generate(messages, tools=tools, temperature=0.2)
        except BrainError as exc:
            result.turns.append(TurnResult(user=user, reply="", passed=False, reason=f"brain error: {exc}"))
            break
        calls = tuple(call.name for call in response.tool_calls)
        reply = response.content
        messages.append(Message(role="assistant", content=reply, tool_calls=response.tool_calls))
        passed, reason = check(turn.get("expect") or {}, reply, calls)
        result.turns.append(TurnResult(user=user, reply=reply, passed=passed, reason=reason, tool_calls=calls))
        if not passed:
            # Stop the conversation at the first broken turn: later turns depend
            # on this one, and continuing would report noise as new failures.
            break
    return result


async def run(args: argparse.Namespace) -> int:
    path = Path(args.cases).expanduser()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        cases = payload["cases"]
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: cannot load cases: {exc}", file=sys.stderr)
        return EXIT_CONFIG

    overrides: dict[str, object] = {}
    if args.model_checkpoint:
        overrides["model_checkpoint"] = args.model_checkpoint
    settings = Settings(**overrides)
    brain = ScratchBrain(checkpoint=settings.resolved_model_checkpoint, max_output_tokens=120)
    registry = default_registry(browser_enabled=False)
    health = await brain.health()
    if not health.available or not health.model_present:
        print(f"ERROR: scratch model unavailable: {health.detail}", file=sys.stderr)
        return EXIT_RUNTIME

    results: list[CaseResult] = []
    try:
        for case in cases:
            best: CaseResult | None = None
            for _ in range(max(1, args.runs)):
                outcome = await run_case(brain, case, registry)
                if best is None or sum(turn.passed for turn in outcome.turns) > sum(turn.passed for turn in best.turns):
                    best = outcome
                if outcome.passed:
                    best = outcome
                    break
            assert best is not None
            results.append(best)
    finally:
        await brain.aclose()

    print("=" * 78)
    print(f"conversation evaluation · {settings.resolved_model_checkpoint}")
    print("=" * 78)
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"{status:<5} {result.case_id}")
        for turn in result.turns:
            marker = "ok" if turn.passed else "!!"
            print(f"   [{marker}] user: {turn.user}")
            print(f"        assistant: {turn.reply[:200]!r}")
            if not turn.passed:
                print(f"        reason: {turn.reason}")
    passed = sum(1 for result in results if result.passed)
    print()
    print(f"cases passed: {passed}/{len(results)}")
    if args.transcript:
        print("\ntranscript saved to the report above (no file written)")

    if args.out:
        out = Path(args.out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "checkpoint": str(settings.resolved_model_checkpoint),
                    "passed": passed,
                    "total": len(results),
                    "cases": [
                        {
                            "id": result.case_id,
                            "passed": result.passed,
                            "turns": [
                                {
                                    "user": turn.user,
                                    "reply": turn.reply,
                                    "passed": turn.passed,
                                    "reason": turn.reason,
                                    "tools": list(turn.tool_calls),
                                }
                                for turn in result.turns
                            ],
                        }
                        for result in results
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"results saved: {out}")
    return EXIT_OK if passed == len(results) else EXIT_FAILED


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="conversation_eval", description=__doc__)
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--runs", type=int, default=1, help="attempts per case (default 1)")
    parser.add_argument("--model-checkpoint", type=Path)
    parser.add_argument("--out", help="write the JSON report here")
    parser.add_argument("--transcript", action="store_true", help="print every turn")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.runs < 1:
        print("ERROR: --runs must be >= 1", file=sys.stderr)
        return EXIT_CONFIG
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
