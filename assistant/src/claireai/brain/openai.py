"""OpenAI Chat Completions adapter.

The wire format here is also what self-hosted OpenAI-compatible servers (vLLM,
LM Studio, llama.cpp's server) speak, so ``CLAIREAI_BRAIN_BASE_URL`` is enough to
point this adapter at a runtime on the owner's own network.

Tool results need a ``tool_call_id``, which the provider-neutral ``Message`` does
not carry. The adapter assigns deterministic ids when it renders the assistant
turn and hands them back in the following tool messages, in call order — the
order the agent emits them in.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping

from claireai.brain.base import BrainError, BrainHealth, BrainResponseError, ToolCall
from claireai.brain.remote import RemoteBrain, messages_for_http, parse_arguments


class OpenAIBrain(RemoteBrain):
    name = "openai"
    provider = "OpenAI"
    default_base_url = "https://api.openai.com/v1"

    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def health(self) -> BrainHealth:
        try:
            payload = await self._request("GET", "/models")
        except BrainError as exc:
            return BrainHealth(False, False, detail=str(exc))
        models = tuple(
            str(item.get("id"))
            for item in payload.get("data") or []
            if isinstance(item, Mapping) and item.get("id")
        )
        present = self.model in models
        detail = (
            f"OpenAI reports {self.model}"
            if present
            else f"OpenAI is reachable but {self.model} is not in its model list"
        )
        return BrainHealth(True, present, models, detail)

    async def generate(self, messages, tools=None, *, temperature=None):  # noqa: ANN001, ANN201
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _wire_messages(messages),
            "max_tokens": self.max_tokens,
        }
        if temperature is not None:
            payload["temperature"] = max(0.0, float(temperature))
        schemas = [
            {
                "type": "function",
                "function": {
                    "name": str(function.get("name")),
                    "description": str(function.get("description") or ""),
                    "parameters": function.get("parameters") or {"type": "object", "properties": {}},
                },
            }
            for schema in tools or ()
            if (function := self._function(schema)).get("name")
        ]
        if schemas:
            payload["tools"] = schemas
            payload["tool_choice"] = "auto"

        data = await self._request("POST", "/chat/completions", payload=payload)
        choices = data.get("choices") or []
        if not choices:
            raise BrainResponseError("OpenAI returned no choices")
        message = choices[0].get("message") or {}
        calls: list[ToolCall] = []
        for index, item in enumerate(message.get("tool_calls") or []):
            function = (item or {}).get("function") or {}
            name = function.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            calls.append(
                ToolCall(
                    name=name,
                    arguments=parse_arguments(function.get("arguments"), provider=self.provider),
                    index=index,
                )
            )
        usage = data.get("usage") or {}
        return self._response(
            content=str(message.get("content") or ""),
            tool_calls=calls,
            model=str(data.get("model") or self.model),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )


def _wire_messages(messages: Iterable[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    call_ids: list[str] = []
    counter = 0
    for message in messages_for_http(messages):
        if message.role == "assistant":
            call_ids = []
            entry: dict[str, Any] = {"role": "assistant", "content": message.content}
            if message.tool_calls:
                entry["tool_calls"] = []
                for call in message.tool_calls:
                    counter += 1
                    call_id = f"call_{counter:04d}"
                    call_ids.append(call_id)
                    entry["tool_calls"].append(
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments, ensure_ascii=False),
                            },
                        }
                    )
            out.append(entry)
        elif message.role == "tool":
            call_id = call_ids.pop(0) if call_ids else f"call_{counter:04d}"
            out.append({"role": "tool", "tool_call_id": call_id, "content": message.content})
        else:
            out.append({"role": message.role, "content": message.content})
    return out
