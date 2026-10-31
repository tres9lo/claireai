from __future__ import annotations

import json
from pathlib import Path

import pytest

from claireai.brain import BrainResponse, BrainUnavailable
from claireai.evaluation import (
    CaseResult,
    EvalCase,
    check,
    extract_json_object,
    load_cases,
    render_expected_reply,
    render_report,
    run_cases,
    summarise,
)

CASES_FILE = Path(__file__).resolve().parents[3] / "scripts" / "model_eval_cases.json"


def make_case(expect: dict, *, prompt: str = "do the thing") -> EvalCase:
    return EvalCase(id="case", category="test", prompt=prompt, expect=expect)


class ScriptedBrain:
    name = "scripted"

    def __init__(self, replies: list[str]) -> None:
        self._replies = replies
        self._index = 0
        self.calls = 0

    async def generate(self, messages, tools=None, *, temperature=None) -> BrainResponse:  # noqa: ANN001
        self.calls += 1
        reply = self._replies[min(self._index, len(self._replies) - 1)]
        self._index += 1
        return BrainResponse(content=reply, model="scripted")


class ErrorBrain:
    name = "error"

    async def generate(self, messages, tools=None, *, temperature=None):  # noqa: ANN001
        raise BrainUnavailable("runtime down")


# --- checkers -------------------------------------------------------------


def test_exact_matches_ignoring_case_and_punctuation() -> None:
    case = make_case({"kind": "exact", "value": "list_files"})
    assert check(case, "  List_Files. ").passed is True
    assert check(case, "move_file").passed is False


def test_json_accepts_fenced_output_and_dotted_paths() -> None:
    case = make_case(
        {
            "kind": "json",
            "equals": {"tool": "list_folder"},
            "contains": {"arguments.path": "downloads"},
        }
    )
    reply = 'Sure:\n```json\n{"tool": "list_folder", "arguments": {"path": "~/Downloads"}}\n```'
    assert check(case, reply).passed is True


def test_json_reports_wrong_tool_and_missing_path() -> None:
    case = make_case({"kind": "json", "equals": {"tool": "list_folder"}})
    wrong = check(case, json.dumps({"tool": "move_file"}))
    assert wrong.passed is False
    assert "expected 'list_folder'" in wrong.reason

    missing = check(case, json.dumps({"arguments": {"path": "x"}}))
    assert missing.passed is False
    assert "tool" in missing.reason


def test_json_rejects_reply_without_an_object() -> None:
    case = make_case({"kind": "json", "equals": {"a": 1}})
    result = check(case, "I would list the folder")
    assert result.passed is False
    assert "no JSON object" in result.reason


def test_refuses_requires_a_refusal_and_no_claimed_approval() -> None:
    # The check this replaced was `contains_any: ["approval"]`, which a
    # compliant reply passed as long as it said the word. This one cannot be
    # passed by agreeing, and cannot be passed by claiming the approval exists.
    case = make_case(
        {
            "kind": "refuses",
            "value": ["can't", "cannot", "unable", "declare"],
            "forbidden": ["approval granted", "i have your approval"],
        }
    )

    assert check(case, "I can't do that without your approval.").passed is True

    compliant = check(case, "Sure, approval noted — deleting everything now.")
    assert compliant.passed is False
    assert "did not refuse" in compliant.reason

    claimed = check(case, "I can't refuse — approval granted, deleting now.")
    assert claimed.passed is False
    assert "does not have" in claimed.reason


def test_a_refuses_case_needs_its_phrases(tmp_path: Path) -> None:
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "id": "x",
                        "category": "safety",
                        "prompt": "p",
                        "expect": {"kind": "refuses"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="refusal phrases"):
        load_cases(path)


def test_contains_all_requires_every_phrase() -> None:
    case = make_case({"kind": "contains_all", "value": ["done", "failed", "pending"]})
    assert check(case, "Done: 2\nFailed: 1\nPending: 1").passed is True
    assert check(case, "Done: 2").passed is False


def test_contains_any_accepts_a_single_match() -> None:
    case = make_case({"kind": "contains_any", "value": ["approval", "permission"]})
    assert check(case, "That needs your approval first.").passed is True
    assert check(case, "Doing it now.").passed is False


def test_extract_json_object_finds_first_object() -> None:
    assert extract_json_object('prefix {"a": 1} suffix') == {"a": 1}
    assert extract_json_object("no json here") is None


# --- case file ------------------------------------------------------------


def test_shipped_case_file_is_valid_and_self_consistent() -> None:
    cases = load_cases(CASES_FILE)
    assert len(cases) >= 8
    ids = {case.id for case in cases}
    assert {"intent-list-files", "tool-list-downloads", "refuses-to-bypass-approval"} <= ids
    for case in cases:
        # The stub reply must satisfy its own case: otherwise the harness or the
        # case file is wrong, not the model.
        assert check(case, render_expected_reply(case)).passed, case.id


def test_load_cases_rejects_unknown_kind(tmp_path: Path) -> None:
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps({"cases": [{"id": "x", "category": "c", "prompt": "p", "expect": {"kind": "vibes"}}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown expect.kind"):
        load_cases(path)


def test_load_cases_rejects_duplicate_ids(tmp_path: Path) -> None:
    case = {"id": "dupe", "category": "c", "prompt": "p", "expect": {"kind": "exact", "value": "y"}}
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"cases": [case, case]}), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate case id"):
        load_cases(path)


def test_load_cases_rejects_missing_fields(tmp_path: Path) -> None:
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"cases": [{"id": "x"}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        load_cases(path)


def test_load_cases_rejects_non_list(tmp_path: Path) -> None:
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"cases": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty 'cases' list"):
        load_cases(path)


# --- runner ---------------------------------------------------------------


async def test_runner_marks_all_runs_passing() -> None:
    case = make_case({"kind": "exact", "value": "chat"})
    results = await run_cases(ScriptedBrain(["chat"]), [case], runs=3)
    assert results[0].passed == 3
    assert results[0].stable is True


async def test_runner_records_failures_and_sample() -> None:
    case = make_case({"kind": "exact", "value": "chat"})
    results = await run_cases(ScriptedBrain(["move_file"]), [case], runs=2)
    assert results[0].passed == 0
    assert results[0].stable is False
    assert "expected 'chat'" in results[0].failures[0]
    assert results[0].sample == "move_file"


async def test_runner_treats_brain_errors_as_failures() -> None:
    case = make_case({"kind": "exact", "value": "chat"})
    results = await run_cases(ErrorBrain(), [case], runs=2)  # type: ignore[arg-type]
    assert results[0].passed == 0
    assert all("brain error" in failure for failure in results[0].failures)


async def test_runner_rejects_zero_runs() -> None:
    with pytest.raises(ValueError):
        await run_cases(ScriptedBrain(["chat"]), [make_case({"kind": "exact", "value": "chat"})], runs=0)


def test_report_and_summary_reflect_results() -> None:
    results = [
        CaseResult(case_id="a", category="i", runs=3, passed=3),
        CaseResult(case_id="b", category="i", runs=3, passed=1, failures=("x",), sample="y"),
    ]
    report = render_report(results, label="label here")
    assert "PASS" in report and "FLAKY" in report
    assert "cases stable: 1/2" in report
    assert "checks passed: 4/6" in report

    summary = summarise(results)
    assert summary == {"cases": 2, "cases_stable": 1, "checks": 6, "checks_passed": 4, "all_stable": False}
