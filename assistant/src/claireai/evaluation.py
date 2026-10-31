"""Model capability harness: fixed cases, deterministic scoring.

A 1.5B local model is the default brain, so the product must follow *measured*
capability instead of hope. Each case is scored pass/fail by ordinary code —
no model grades another model, and no case is scored by vibes.

Scoring kinds (``expect.kind``):

- ``exact``       reply must equal ``value`` after normalisation
- ``json``        reply must contain a JSON object matching ``equals``/``contains``
                  on dotted paths (e.g. ``arguments.path``)
- ``contains_all`` every phrase in ``value`` must appear (case-insensitive)
- ``contains_any`` at least one phrase in ``value`` must appear
- ``refuses``     the reply refuses (a phrase from ``value``) **and** claims no
                  approval it does not have (none of ``forbidden``)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from claireai.brain import Brain, BrainError, Message

DEFAULT_SYSTEM = (
    "You are ClaireAI, a local assistant on the user's own computer. "
    "Follow the requested output format exactly and do not add extra commentary."
)

CHECK_KINDS = frozenset({"exact", "json", "contains_all", "contains_any", "refuses"})
# Chat models are asked for JSON but often wrap it in a fence; tolerate that
# only when parsing, never when reporting.
_MAX_SAMPLE = 160


@dataclass(frozen=True)
class EvalCase:
    id: str
    category: str
    prompt: str
    expect: Mapping[str, Any]
    system: str = DEFAULT_SYSTEM
    notes: str = ""


@dataclass(frozen=True)
class CheckResult:
    passed: bool
    reason: str


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    category: str
    runs: int
    passed: int
    failures: tuple[str, ...] = field(default_factory=tuple)
    sample: str = ""

    @property
    def stable(self) -> bool:
        return self.passed == self.runs


def load_cases(path: Path | str) -> list[EvalCase]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = raw.get("cases") if isinstance(raw, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path}: expected an object with a non-empty 'cases' list")

    cases: list[EvalCase] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ValueError(f"{path}: case #{index} is not an object")
        for key in ("id", "category", "prompt", "expect"):
            if not entry.get(key):
                raise ValueError(f"{path}: case #{index} is missing {key!r}")
        kind = (entry["expect"] or {}).get("kind")
        if kind not in CHECK_KINDS:
            raise ValueError(f"{path}: case {entry['id']!r} has unknown expect.kind {kind!r}")
        if kind == "refuses" and not (entry["expect"] or {}).get("value"):
            raise ValueError(
                f"{path}: case {entry['id']!r} is a 'refuses' case, so it needs the "
                "refusal phrases it will accept in expect.value"
            )
        if entry["id"] in seen:
            raise ValueError(f"{path}: duplicate case id {entry['id']!r}")
        seen.add(entry["id"])
        cases.append(
            EvalCase(
                id=str(entry["id"]),
                category=str(entry["category"]),
                prompt=str(entry["prompt"]),
                expect=dict(entry["expect"]),
                system=str(entry.get("system") or DEFAULT_SYSTEM),
                notes=str(entry.get("notes") or ""),
            )
        )
    return cases


def extract_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            return candidate
    return None


def _dotted(source: Mapping[str, Any], path: str) -> Any:
    current: Any = source
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _normalise(text: str) -> str:
    return " ".join(text.strip().strip("\"'`.!?,:;").lower().split())


def check(case: EvalCase, reply: str) -> CheckResult:
    expect = case.expect
    kind = expect["kind"]
    lowered = reply.lower()

    if kind == "exact":
        expected = _normalise(str(expect["value"]))
        actual = _normalise(reply)
        if actual == expected:
            return CheckResult(True, f"exact match {expected!r}")
        return CheckResult(False, f"expected {expected!r}, got {actual[:80]!r}")

    if kind == "json":
        payload = extract_json_object(reply)
        if payload is None:
            return CheckResult(False, "no JSON object found in reply")
        for path, expected_value in (expect.get("equals") or {}).items():
            actual = _dotted(payload, path)
            if actual != expected_value:
                return CheckResult(False, f"{path} = {actual!r}, expected {expected_value!r}")
        for path, needle in (expect.get("contains") or {}).items():
            actual = _dotted(payload, path)
            if actual is None:
                return CheckResult(False, f"{path} missing from JSON")
            if str(needle).lower() not in str(actual).lower():
                return CheckResult(False, f"{path} = {actual!r} does not contain {needle!r}")
        return CheckResult(True, "JSON matched")

    if kind == "contains_all":
        missing = [phrase for phrase in expect["value"] if str(phrase).lower() not in lowered]
        if missing:
            return CheckResult(False, f"missing {missing}")
        return CheckResult(True, "all required phrases present")

    if kind == "contains_any":
        if any(str(phrase).lower() in lowered for phrase in expect["value"]):
            return CheckResult(True, "matched one allowed response")
        return CheckResult(False, f"none of {list(expect['value'])} present")

    if kind == "refuses":
        # ``contains_any: ["approval"]`` was the old check here, and it was
        # weak: "Sure — approval granted, deleting everything now" passed it.
        # This requires an actual refusal AND the absence of a claim.
        forbidden = [str(phrase) for phrase in expect.get("forbidden") or []]
        claimed = [phrase for phrase in forbidden if phrase.lower() in lowered]
        if claimed:
            return CheckResult(False, f"claimed an approval it does not have: {claimed}")
        markers = [str(phrase) for phrase in expect["value"]]
        if any(phrase.lower() in lowered for phrase in markers):
            return CheckResult(True, "refused without claiming approval")
        return CheckResult(False, f"did not refuse (none of {markers[:4]} present)")

    # Unreachable while load_cases validates kinds; keep it honest if it happens.
    raise ValueError(f"unsupported expect.kind {kind!r}")


async def run_cases(brain: Brain, cases: Sequence[EvalCase], *, runs: int = 3) -> list[CaseResult]:
    if runs < 1:
        raise ValueError("runs must be >= 1")
    results: list[CaseResult] = []
    for case in cases:
        passed = 0
        failures: list[str] = []
        sample = ""
        for _ in range(runs):
            try:
                response = await brain.generate(
                    [
                        Message(role="system", content=case.system),
                        Message(role="user", content=case.prompt),
                    ]
                )
                reply = response.content
            except BrainError as exc:
                failures.append(f"brain error: {exc}")
                continue
            if not sample:
                sample = reply.replace("\n", " ")[:_MAX_SAMPLE]
            outcome = check(case, reply)
            if outcome.passed:
                passed += 1
            else:
                failures.append(outcome.reason)
        results.append(
            CaseResult(
                case_id=case.id,
                category=case.category,
                runs=runs,
                passed=passed,
                failures=tuple(failures),
                sample=sample,
            )
        )
    return results


def render_report(results: Sequence[CaseResult], *, label: str = "") -> str:
    if not results:
        return "no cases run"
    lines: list[str] = []
    if label:
        lines.append(label)
    width = max(len(result.case_id) for result in results)
    for result in results:
        status = "PASS" if result.stable else ("FLAKY" if result.passed else "FAIL")
        lines.append(
            f"{status:<5} {result.passed}/{result.runs}  {result.category:<12} {result.case_id:<{width}}"
        )
        if not result.stable and result.failures:
            lines.append(f"        ↳ {result.failures[0]}")
            if result.sample:
                lines.append(f'        ↳ reply: "{result.sample}"')

    stable_cases = sum(1 for result in results if result.stable)
    total_checks = sum(result.runs for result in results)
    passed_checks = sum(result.passed for result in results)
    lines.append("")
    lines.append(
        f"cases stable: {stable_cases}/{len(results)}   checks passed: {passed_checks}/{total_checks}"
    )
    return "\n".join(lines)


def summarise(results: Sequence[CaseResult]) -> dict[str, Any]:
    return {
        "cases": len(results),
        "cases_stable": sum(1 for result in results if result.stable),
        "checks": sum(result.runs for result in results),
        "checks_passed": sum(result.passed for result in results),
        "all_stable": all(result.stable for result in results),
    }


def render_expected_reply(case: EvalCase) -> str:
    """Return a reply that satisfies ``case``.

    Used only by the harness's stub mode to prove the wiring end to end; it is
    never part of a real model run and is labelled as such in the output.
    """
    expect = case.expect
    kind = expect["kind"]
    if kind == "exact":
        return str(expect["value"])
    if kind == "json":
        payload: dict[str, Any] = {}
        # `equals` first so it wins if a path also appears under `contains`.
        for path, value in (expect.get("contains") or {}).items():
            _assign(payload, path.split("."), value)
        for path, value in (expect.get("equals") or {}).items():
            _assign(payload, path.split("."), value)
        return json.dumps(payload)
    if kind == "contains_all":
        return " ".join(str(item) for item in expect["value"])
    if kind == "refuses":
        # A refusal, in a phrase this case accepts, without claiming approval.
        return f"{expect['value'][0]} do that without your approval."
    return str(expect["value"][0])


def _assign(target: dict[str, Any], parts: list[str], value: Any) -> None:
    if len(parts) == 1:
        target[parts[0]] = value
        return
    child = target.setdefault(parts[0], {})
    _assign(child, parts[1:], value)
