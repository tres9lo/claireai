"""Anthropic Messages API adapter.

Two differences from the OpenAI shape are handled here rather than in the agent:

- the system prompt is a top-level field, not a message;
- a tool result is a content block inside a **user** message, so consecutive
  results are merged into one turn, which is what the API expects.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from claireai.brain.base import BrainError, BrainHealth, ToolCall
from claireai.brain.remote import RemoteBrain, messages_for_http, parse_arguments

ANTHROPIC_VERSION = "2023-06-01"


class AnthropicBrain(RemoteBrain):
    name = "anthropic"
    provider = "Anthropic"
    default_base_url = "https://api.anthropic.com/v1"

    def headers(self) -> dict[str, str]:
        return {
            "x-api-key": str(self.api_key),
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

    async def health(self) -> BrainHealth:
        try:
            payload = await self._request("GET", "/models", params={"limit": 50})
        except BrainError as exc:
            return BrainHealth(False, False, detail=str(exc))
        models = tuple(
            str(item.get("id"))
            for item in payload.get("data") or []
            if isinstance(item, Mapping) and item.get("id")
        )
        present = self.model in models
        detail = (
            f"Anthropic reports {self.model}"
            if present
            else f"Anthropic is reachable but {self.model} is not in its model list"
        )
        return BrainHealth(True, present, models, detail)

    async def generate(self, messages, tools=None, *, temperature=None):  # noqa: ANN001, ANN201
        system, wire = _wire_messages(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": wire,
        }
        if system:
            payload["system"] = system
        if temperature is not None:
            # Anthropic accepts 0.0–1.0; the runtime can ask for 2.0.
            payload["temperature"] = min(1.0, max(0.0, float(temperature)))
        schemas = [
            {
                "name": str(function.get("name")),
                "description": str(function.get("description") or ""),
                "input_schema": function.get("parameters") or {"type": "object", "properties": {}},
            }
            for schema in tools or ()
            if (function := self._function(schema)).get("name")
        ]
        if schemas:
            payload["tools"] = schemas

        data = await self._request("POST", "/messages", payload=payload)
        blocks = data.get("content") or []
        text = "".join(
            str(block.get("text") or "") for block in blocks if block.get("type") == "text"
        )
        calls: list[ToolCall] = []
        for block in blocks:
            if block.get("type") != "tool_use":
                continue
            name = block.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            calls.append(
                ToolCall(
                    name=name,
                    arguments=parse_arguments(block.get("input"), provider=self.provider),
                    index=len(calls),
                )
            )
        usage = data.get("usage") or {}
        return self._response(
            content=text,
            tool_calls=calls,
            model=str(data.get("model") or self.model),
            prompt_tokens=usage.get("input_tokens"),
            completion_tokens=usage.get("output_tokens"),
        )


def _wire_messages(messages: Iterable[Any]) -> tuple[str, list[dict[str, Any]]]:
    system_parts: list[str] = []
    wire: list[dict[str, Any]] = []
    call_ids: list[str] = []
    counter = 0
    for message in messages_for_http(messages):
        if message.role == "system":
            if message.content.strip():
                system_parts.append(message.content)
        elif message.role == "assistant":
            call_ids = []
            blocks: list[dict[str, Any]] = []
            if message.content:
                blocks.append({"type": "text", "text": message.content})
            for call in message.tool_calls:
                counter += 1
                call_id = f"toolu_{counter:04d}"
                call_ids.append(call_id)
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call_id,
                        "name": call.name,
                        "input": call.arguments,
                    }
                )
            wire.append({"role": "assistant", "content": blocks or [{"type": "text", "text": ""}]})
        elif message.role == "tool":
            call_id = call_ids.pop(0) if call_ids else f"toolu_{counter:04d}"
            block = {"type": "tool_result", "tool_use_id": call_id, "content": message.content}
            if wire and wire[-1]["role"] == "user" and all(
                item.get("type") == "tool_result" for item in wire[-1]["content"]
            ):
                wire[-1]["content"].append(block)
            else:
                wire.append({"role": "user", "content": [block]})
        else:
            wire.append({"role": "user", "content": [{"type": "text", "text": message.content}]})
    return "\n\n".join(system_parts), wire
