"""Read the tool calls a small model actually writes.

The scratch model is trained on exactly one shape, rendered for it in the prompt
block:

    <tool_call>{"name":"list_folder","arguments":{"path":"~/Downloads"}}</tool_call>

At three million parameters and a sampling temperature, it also produces the
neighbouring shapes: a dropped closing tag, a code fence, single quotes, a
trailing comma, ``"tool"`` where ``"name"`` belongs, arguments sent as a JSON
string, or a call wrapped in a sentence. Only the first shape was ever accepted,
which turned every one of those into a failed task.

Repairs here are **formatting only**. This module never widens permission: a name
that was not offered is rejected, and the arguments it returns are validated
against the tool's pydantic model and the folder allow-list by the agent exactly
as before. A tolerant reader must not become a more permissive runtime.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from claireai.brain.base import ToolCall

TAG_OPEN = "<tool_call>"
TAG_CLOSE = "</tool_call>"

_NAME_KEYS = ("name", "tool", "tool_name", "function", "action", "action_name")
_ARGUMENT_KEYS = ("arguments", "args", "parameters", "parameters_json", "params", "input")
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_BARE_KEY = re.compile(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:")


@dataclass(frozen=True)
class ParsedToolCalls:
    """What could be read, and whether the model was clearly trying to call."""

    calls: tuple[ToolCall, ...] = ()
    saw_intent: bool = False
    problem: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.calls)


def looks_like_tool_call(text: str) -> bool:
    """Was this reply an attempted call that produced no usable call?

    Deliberately narrow: the text must *start* with the tag or a JSON object, so
    an ordinary answer that quotes the format in passing is not mistaken for a
    broken call.
    """
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.lower().startswith("<tool_call"):
        return True
    if stripped.startswith("{") and stripped.endswith("}"):
        return any(f'"{key}"' in stripped for key in _NAME_KEYS)
    return False


def parse_tool_calls(text: str, offered: Iterable[str] = ()) -> ParsedToolCalls:
    """Extract every readable call from one assistant message."""
    names = tuple(str(name).strip() for name in offered if str(name).strip())
    chunks = list(_payload_chunks(text))
    saw_intent = bool(chunks) or looks_like_tool_call(text)
    calls: list[ToolCall] = []
    problems: list[str] = []
    for chunk in chunks:
        payload = _loads(chunk)
        if payload is None:
            problems.append("a payload that is not valid JSON")
            continue
        found, issues = _calls_from_payload(payload, names)
        calls.extend(found)
        problems.extend(issues)
    if calls:
        return ParsedToolCalls(tuple(calls), saw_intent)
    return ParsedToolCalls((), saw_intent, "; ".join(dict.fromkeys(problems)))


def _payload_chunks(text: str) -> Iterable[str]:
    """Every candidate payload, most explicit source first."""
    found = False
    index = 0
    while True:
        start = text.find(TAG_OPEN, index)
        if start < 0:
            break
        body_start = start + len(TAG_OPEN)
        end = text.find(TAG_CLOSE, body_start)
        body = text[body_start : end if end >= 0 else len(text)]
        yield _first_object(body) or body.strip()
        found = True
        index = (end + len(TAG_CLOSE)) if end >= 0 else len(text)
    if found:
        return
    for match in _FENCE.finditer(text):
        body = match.group(1).strip()
        if body:
            yield _first_object(body) or body
            found = True
    if found:
        return
    # A bare payload is accepted only when the whole reply is one. Pulling JSON
    # out of the middle of prose would turn a quoted example into an action.
    stripped = text.strip()
    if stripped.startswith(("{", "[")):
        candidate = _first_object(stripped)
        if candidate:
            yield candidate


def _first_object(text: str) -> str | None:
    """The first balanced ``{...}`` or ``[...]`` run, string- and escape-aware."""
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        while start >= 0:
            depth = 0
            quote = ""
            escaped = False
            for index in range(start, len(text)):
                char = text[index]
                if quote:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == quote:
                        quote = ""
                    continue
                if char in "\"'":
                    quote = char
                elif char == opener:
                    depth += 1
                elif char == closer:
                    depth -= 1
                    if depth == 0:
                        return text[start : index + 1]
            start = text.find(opener, start + 1)
    return None


def _loads(text: str) -> Any:
    """JSON first, then the small set of near-JSON shapes models emit."""
    attempts = (
        text,
        _BARE_KEY.sub(r'\1"\2":', text),
        re.sub(r",\s*([}\]])", r"\1", text),
    )
    for attempt in attempts:
        try:
            return json.loads(attempt)
        except ValueError:
            continue
    # Python literals cover single quotes, ``True``/``None`` and tuples, which
    # ``ast`` parses without evaluating anything.
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return None
    return value if isinstance(value, (dict, list)) else None


def _match_name(name: str, offered: Sequence[str]) -> str | None:
    """The offered tool this name means, or ``None``.

    Only normalisation of punctuation and case is allowed: a model that invents a
    name must not be mapped onto a real tool that happens to look similar.
    """
    if not offered:
        return name
    cleaned = name.strip()
    if cleaned in offered:
        return cleaned
    folded = cleaned.casefold()
    for candidate in offered:
        if candidate.casefold() == folded:
            return candidate
    normalised = re.sub(r"[\s\-]+", "_", folded)
    for candidate in offered:
        if re.sub(r"[\s\-]+", "_", candidate.casefold()) == normalised:
            return candidate
    return None


def _calls_from_payload(payload: Any, offered: Sequence[str]) -> tuple[list[ToolCall], list[str]]:
    if isinstance(payload, list):
        calls: list[ToolCall] = []
        problems: list[str] = []
        for item in payload:
            found, issues = _calls_from_payload(item, offered)
            calls.extend(found)
            problems.extend(issues)
        return calls, problems
    if not isinstance(payload, dict):
        return [], ["a payload that is not a JSON object"]
    if isinstance(payload.get("tool_calls"), list):
        return _calls_from_payload(payload["tool_calls"], offered)
    if isinstance(payload.get("function"), dict):
        # OpenAI's own wire shape nests the call inside "function".
        payload = payload["function"]

    name = next(
        (
            payload[key]
            for key in _NAME_KEYS
            if isinstance(payload.get(key), str) and payload[key].strip()
        ),
        None,
    )
    if name is None:
        return [], ["a payload without a tool name"]
    matched = _match_name(name, offered)
    if matched is None:
        return [], [f"unknown tool {name!r}"]

    arguments = next((payload[key] for key in _ARGUMENT_KEYS if key in payload), {})
    if isinstance(arguments, str):
        arguments = _loads(arguments)
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        return [], [f"arguments for {matched!r} were not an object"]
    return [ToolCall(name=matched, arguments=arguments)], []
