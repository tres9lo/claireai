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
