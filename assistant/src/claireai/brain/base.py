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
class Message:
    role: Role
    content: str

    def as_wire(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class BrainResponse:
    content: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


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
