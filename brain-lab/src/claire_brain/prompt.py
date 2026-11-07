"""The one prompt format shared by training and inference.

A model only learns the format it is trained on.  Keeping the renderer here, in
the package both the trainer and the assistant import, means a change to the
separator or the tool block cannot silently make training and inference disagree.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping, Sequence


def _render_call(call: Any) -> str:
    name = str(_field(call, "name", "") or "")
    arguments = _field(call, "arguments", {}) or {}
    payload = json.dumps({"name": name, "arguments": arguments}, separators=(",", ":"), ensure_ascii=False)
    return f"<tool_call>{payload}</tool_call>"


def _field(message: Any, name: str, default: Any = None) -> Any:
    if isinstance(message, Mapping):
        return message.get(name, default)
    return getattr(message, name, default)


def tool_block(tools: Iterable[Mapping[str, Any]]) -> str:
    """Compact, token-cheap description of the tools offered for this request."""
    lines: list[str] = []
    for schema in tools:
        function = schema.get("function", schema) if isinstance(schema, Mapping) else {}
        name = str(function.get("name") or "").strip()
        if not name:
            continue
        description = " ".join(str(function.get("description") or "").split())
        parameters = function.get("parameters") or {}
        properties = parameters.get("properties") if isinstance(parameters, Mapping) else {}
        required = parameters.get("required") if isinstance(parameters, Mapping) else None
        # Required arguments only: optional knobs (limit, sort, max_bytes) cost
        # prompt tokens on every step and have safe defaults.  The model can
        # still pass them, and the runtime validates them normally.
        names = list(required) if isinstance(required, list) and required else list(properties or {})
        arguments = ", ".join(str(name) for name in names)
        if len(description) > 90:
            description = description[:87].rsplit(" ", 1)[0] + "..."
        lines.append(f"- {name}({arguments}): {description}")
    if not lines:
        return ""
    return (
        "TOOLS: You may call these tools:\n"
        + "\n".join(lines)
        + "\nTo call a tool reply exactly: <tool_call>{\"name\":\"tool_name\","
        "\"arguments\":{...}}</tool_call>\nOtherwise answer normally."
    )


def render_prompt(
    system: str | None,
    messages: Sequence[Any],
    tools: Iterable[Mapping[str, Any]] | None = None,
) -> str:
    """Render one prompt ending in ``ASSISTANT: ``.

    ``messages`` may be ``Message`` objects or plain mappings.  A tool result is
    rendered as ``TOOL <name>: <content>`` so the model can tell it apart from a
    user turn.
    """
    system_text = (system or "").strip()
    block = tool_block(tools or ())
    if block:
        system_text = f"{system_text}\n{block}" if system_text else block
    parts = [f"SYSTEM: {system_text}"]
    for message in messages:
        role = str(_field(message, "role", "user"))
        content = str(_field(message, "content", "") or "")
        if role == "assistant":
            calls = _field(message, "tool_calls", ()) or ()
            if calls:
                rendered = "".join(_render_call(call) for call in calls)
                content = f"{content} {rendered}".strip() if content else rendered
            parts.append(f"ASSISTANT: {content}")
        elif role == "tool":
            name = str(_field(message, "tool_name", "") or "tool")
            parts.append(f"TOOL {name}: {content}")
        elif role == "system":
            # A second system message is appended rather than replacing the first;
            # the runtime only sends one, but training data may carry one.
            parts.append(f"SYSTEM: {content}")
        else:
            parts.append(f"USER: {content}")
    # The trailing space is part of the prompt: the target text is the reply
    # itself, so training and inference agree on where the reply starts.
    parts.append("ASSISTANT: ")
    return "\n".join(parts)
