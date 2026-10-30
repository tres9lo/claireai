"""Provider-neutral brain interface.

Agents and the CLI talk to this module only. Nothing outside ``claireai.brain``
may import a specific runtime (Ollama today, the from-scratch model later), so
swapping the brain never touches the agent layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Protocol, Sequence, runtime_checkable

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation proposed by the model.

    The model only ever *proposes*: the agent runtime validates the name,
    arguments and risk level before anything runs (roadmap 6.2, 7.1).
    """

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    index: int = 0

    def as_wire(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {"index": self.index, "name": self.name, "arguments": self.arguments},
        }


@dataclass(frozen=True)
class Message:
    """One chat message in the provider-neutral shape.

    ``tool_calls`` is set on an assistant message that proposed calls, and
    ``tool_name`` on a ``tool`` message carrying one result
    (https://docs.ollama.com/capabilities/tool-calling, checked 2026-10-30).
    """

    role: Role
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_name: str | None = None

    def as_wire(self) -> dict[str, Any]:
        wire: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            wire["tool_calls"] = [call.as_wire() for call in self.tool_calls]
        if self.tool_name:
            wire["tool_name"] = self.tool_name
        return wire


@dataclass(frozen=True)
class BrainResponse:
    content: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True)
class BrainHealth:
    """Distinguishes "runtime reachable" from "configured model is present"."""

    available: bool
    model_present: bool
    models: tuple[str, ...] = field(default_factory=tuple)
    detail: str = ""


class BrainError(Exception):
    """Base class for brain failures the caller is expected to handle."""


class BrainUnavailable(BrainError):
    """The local model runtime is not reachable at the configured address."""


class BrainResponseError(BrainError):
    """The runtime answered, but rejected the request or returned junk."""


@runtime_checkable
class Brain(Protocol):
    name: str

    async def health(self) -> BrainHealth:
        ...

    async def generate(
        self,
        messages: Sequence[Message],
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> BrainResponse:
        """Produce one assistant message.

        ``tools`` is part of the contract from the start so the agent runtime can
        pass tool schemas later; adapters may ignore it while unused.
        """
        ...
