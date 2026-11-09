"""The corpus prompt must be the runtime prompt.

This is a regression test with a measurement behind it. The chat-v3 corpus
carried a hand-written system sentence while ``claireai.agent`` sent a different
one. On 2026-11-06 the same weights scored 3/138 tool-call checks through the
runtime prompt and emitted correct calls when the corpus sentence was used
instead — the assistant was being asked a prompt it had never seen.

Keeping the corpus sentence imported from the runtime is what makes that
impossible; these assertions are what notice if someone reintroduces a copy.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from claireai.agent import system_message
from claireai.permissions import Sandbox

BUILDER = Path(__file__).resolve().parents[3] / "brain-lab" / "build_chat_dataset.py"


def _builder():  # noqa: ANN202
    spec = importlib.util.spec_from_file_location("build_chat_dataset_under_test", BUILDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_corpus_sentence_is_the_runtime_sentence() -> None:
    module = _builder()
    assert module.RUNTIME_SYSTEM == system_message(Sandbox(roots=())).content


def test_folder_variants_keep_the_runtime_prefix_and_one_clause() -> None:
    module = _builder()
    prefix = system_message(Sandbox(roots=())).content.split(" Allowed folders:")[0]
    seen = set()
    for _ in range(40):
        system = module.assistant_system()
        seen.add(system)
        assert system.startswith(prefix)
        assert system.count("Allowed folders:") == 1
    assert len(seen) > 1, "the folder tail never varied"


def test_the_empty_folder_clause_is_the_runtime_clause() -> None:
    """The "no folders" sentence must be the one the runtime prints, not a copy.

    It used to say "none configured.", a sentence ``Sandbox.describe`` never
    returns.  The tool harness evaluates with an empty sandbox, so on 2026-11-08
    chat-v10 was scored on a prompt that appeared in no training record and
    answered it with a refusal.  Measured afterwards: 33/165 checks under that
    prompt against 71/165 under the folder clause the corpus actually teaches.
    """
    module = _builder()
    runtime_clause = system_message(Sandbox(roots=())).content.split(" Allowed folders:", 1)[1]
    assert runtime_clause in module.FOLDER_TAILS
    assert not any("none configured" in tail for tail in module.FOLDER_TAILS)


def test_a_record_that_calls_a_tool_never_claims_no_folders() -> None:
    """A call needs a folder the model may use, or the record teaches nonsense."""
    module = _builder()
    empty_clause = system_message(Sandbox(roots=())).content.split(" Allowed folders:", 1)[1]
    for _ in range(40):
        assert empty_clause not in module.tool_system(["list_folder"])


def test_a_grounded_record_keeps_its_own_folder_clause() -> None:
    module = _builder()
    grounded = module.tool_system(
        ["move_file"], extra="Allowed folders: /home/owner/Downloads.\ncontents"
    )
    assert grounded.count("Allowed folders:") == 1
    assert "/home/owner/Downloads" in grounded
    assert module.tool_system(["list_folder"]).count("Allowed folders:") == 1
