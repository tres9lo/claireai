"""The training prompt and the runtime prompt must be the same format."""

from __future__ import annotations

import pytest

prompt_module = pytest.importorskip("claire_brain.prompt")
from claire_brain.prompt import render_prompt, tool_block  # noqa: E402

from claireai.brain import Message, ScratchBrain, ToolCall  # noqa: E402


def test_prompt_ends_where_the_reply_starts() -> None:
    prompt = render_prompt("You are ClaireAI.", [{"role": "user", "content": "hello"}])
    assert prompt.endswith("ASSISTANT: ")
    assert "SYSTEM: You are ClaireAI." in prompt
    assert "USER: hello" in prompt


def test_tool_block_lists_required_arguments_and_the_call_format() -> None:
    schema = {
        "type": "function",
        "function": {
            "name": "move_file",
            "description": "Move a file or folder; never overwrites.",
            "parameters": {
                "properties": {
                    "source": {"type": "string"},
                    "destination": {"type": "string"},
                    "overwrite": {"type": "boolean"},
                },
                "required": ["source", "destination"],
            },
        },
    }
    block = tool_block([schema])
    assert "move_file(source, destination):" in block
    assert "overwrite," not in block
    assert "<tool_call>" in block


def test_an_assistant_tool_call_is_rendered_as_text() -> None:
    call = ToolCall(name="list_folder", arguments={"path": "~/Downloads"})
    prompt = render_prompt(
        "You are ClaireAI.",
        [
            {"role": "user", "content": "what is in Downloads"},
            {"role": "assistant", "content": "", "tool_calls": (call,)},
            {"role": "tool", "tool_name": "list_folder", "content": "report.pdf"},
        ],
    )
    assert '<tool_call>{"name":"list_folder","arguments":{"path":"~/Downloads"}}</tool_call>' in prompt
    assert "TOOL list_folder: report.pdf" in prompt


def test_clean_removes_a_prompt_echo_and_stops_at_a_turn_boundary() -> None:
    assert ScratchBrain._clean("ASSISTANT: Hello there.") == "Hello there."
    assert ScratchBrain._clean("Hello.\nUSER: another question") == "Hello."
    assert ScratchBrain._clean("Hello.\nSYSTEM: next") == "Hello."


def test_messages_and_tool_results_keep_their_roles() -> None:
    prompt = render_prompt(
        "system",
        [
            Message(role="user", content="one"),
            Message(role="assistant", content="two"),
            Message(role="tool", content="three", tool_name="list_folder"),
        ],
    )
    assert "USER: one" in prompt
    assert "ASSISTANT: two" in prompt
    assert "TOOL list_folder: three" in prompt
