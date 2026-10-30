"""Tool registry (roadmap 6.2).

A tool is a name, a description, a JSON schema for its inputs and a risk level.
``safe`` tools are read-only and run freely; anything else is refused until the
approval flow exists (see :mod:`claireai.permissions`).

Arguments are validated with the tool's own pydantic model *before* the handler
sees them, so a model cannot smuggle unexpected keys or wrong types into a tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

from pydantic import BaseModel, ValidationError

from claireai.permissions import RISK_LEVELS, Sandbox

class ToolError(Exception):
    """An expected failure the model may recover from (bad path, wrong type)."""


class UnknownTool(ToolError):
    """The model asked for a tool that is not registered."""


@dataclass(frozen=True)
class ToolContext:
    """Everything a tool is allowed to know about the outside world."""

    sandbox: Sandbox
    dry_run: bool = False


@dataclass(frozen=True)
class ToolResult:
    text: str
    detail: Mapping[str, Any] = field(default_factory=dict)


Handler = Callable[[ToolContext, BaseModel], ToolResult]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    risk: str
    args_model: type[BaseModel]
    handler: Handler

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("tool name must not be empty")
        if self.risk not in RISK_LEVELS:
            raise ValueError(f"tool {self.name!r} has unknown risk level {self.risk!r}")
        if not issubclass(self.args_model, BaseModel):
            raise TypeError(f"tool {self.name!r} needs a pydantic args model")

    @property
    def parameters(self) -> dict[str, Any]:
        return self.args_model.model_json_schema()

    def schema(self) -> dict[str, Any]:
        """The wire shape Ollama expects in ``tools``."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def validate(self, arguments: Mapping[str, Any]) -> BaseModel:
        """Validate arguments, turning pydantic's error into a model-readable one.

        ``null`` for an optional parameter means "unspecified" in every tool
        API models are trained on, so it is dropped before validation and the
        declared default applies. Observed for real: ``qwen2.5:1.5b`` sends
        ``"limit": null`` rather than omitting the key. Dropping a null never
        widens permission: a null ``path`` still fails as missing.
        """
        cleaned = {key: value for key, value in arguments.items() if value is not None}
        try:
            return self.args_model.model_validate(cleaned)
        except ValidationError as exc:
            raise ToolError(f"invalid arguments for {self.name!r}: {exc}") from exc


class ToolRegistry:
    """Name → tool, with the invariant that names are unique."""

    def __init__(self, tools: Iterable[Tool]) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"duplicate tool name {tool.name!r}")
            self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            available = ", ".join(sorted(self._tools)) or "(none)"
            raise UnknownTool(f"unknown tool {name!r}; available: {available}") from None

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self):  # noqa: ANN204
        return iter(self._tools.values())

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ToolRegistry({self.names()!r})"
