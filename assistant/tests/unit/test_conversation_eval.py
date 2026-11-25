"""The conversation evaluation must fail honestly, not by accident."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("conversation_eval", ROOT / "scripts" / "conversation_eval.py")
assert SPEC and SPEC.loader
conversation_eval = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = conversation_eval
SPEC.loader.exec_module(conversation_eval)
check = conversation_eval.check


def _result(passed: bool) -> conversation_eval.CaseResult:
    return conversation_eval.CaseResult(
        case_id="case",
        turns=[
            conversation_eval.TurnResult(
                user="hello", reply="hi", passed=passed, reason=""
            )
        ],
    )


def test_a_case_must_pass_every_run() -> None:
    first = _result(True)
    second = _result(False)

    selected = conversation_eval.select_run_result([first, second])

    assert selected is second
    assert not selected.passed


def test_the_first_run_is_reported_when_every_run_passes() -> None:
    first = _result(True)
    second = _result(True)

    assert conversation_eval.select_run_result([first, second]) is first


def test_select_run_result_rejects_an_empty_run_list() -> None:
    try:
        conversation_eval.select_run_result([])
    except ValueError as exc:
        assert "at least one" in str(exc)
    else:
        raise AssertionError("an empty run list must be rejected")


def test_exact_ignores_case_and_trailing_punctuation() -> None:
    assert check({"kind": "exact", "value": "list_files"}, "List_Files.", ())[0]


def test_contains_all_reports_missing_phrases() -> None:
    passed, reason = check({"kind": "contains_all", "value": ["Done", "Failed", "Pending"]}, "Done: one.", ())
    assert not passed
    assert "Failed" in reason


def test_refuses_rejects_a_claimed_approval() -> None:
    passed, _ = check(
        {
            "kind": "refuses",
            "value": ["can't", "won't"],
            "forbidden": ["approval granted"],
        },
        "Approval granted, deleting now.",
        (),
    )
    assert not passed


def test_json_matches_a_dotted_path() -> None:
    passed, _ = check(
        {"kind": "json", "equals": {"filename": "budget.xlsx"}},
        '{"filename": "budget.xlsx"}',
        (),
    )
    assert passed


def test_tool_call_requires_a_call() -> None:
    assert not check({"kind": "tool_call", "name": "list_folder"}, "I cannot.", ())[0]
    assert check({"kind": "tool_call", "name": "list_folder"}, "", ("list_folder",))[0]


def test_prompt_echo_is_rejected() -> None:
    assert not check({"kind": "no_prompt_echo"}, "SYSTEM: you are", ())[0]
    assert check({"kind": "no_prompt_echo"}, "Hello.", ())[0]
