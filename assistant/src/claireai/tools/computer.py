"""Computer-control tool backed by :mod:`claireai.cua`.

The registry only offers this tool when the owner has enabled at least one of
the ``cursor``/``keyboard`` capabilities. The handler then checks the capability
for the *specific* action, so enabling only one of them cannot smuggle in the
other. The tool is ``dangerous``: the agent always asks before it runs.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from claireai.cua import CuaDriver, CuaError
from claireai.permissions import PermissionDenied, check_capability
from claireai.tools.base import ToolContext, ToolError, ToolResult

ACTION_CAPABILITY = {
    "move": "cursor",
    "click": "cursor",
    "scroll": "cursor",
    "type": "keyboard",
    "key": "keyboard",
}


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ComputerControlArgs(_Args):
    action: Literal["move", "click", "scroll", "type", "key"] = Field(
        description="The desktop action to perform."
    )
    x: int | None = Field(default=None, ge=0, le=100_000, description="Screen x pixel.")
    y: int | None = Field(default=None, ge=0, le=100_000, description="Screen y pixel.")
    text: str | None = Field(
        default=None,
        max_length=2_000,
        description="Text to type; never put passwords or API keys here.",
    )
    key: str | None = Field(
        default=None,
        max_length=80,
        description="Key or chord, e.g. 'enter', 'ctrl+s'.",
    )
    amount: int = Field(default=3, ge=-20, le=20, description="Scroll steps; positive is up.")
    button: Literal["left", "middle", "right"] = "left"
    clicks: int = Field(default=1, ge=1, le=3)


def computer_control(context: ToolContext, arguments: ComputerControlArgs) -> ToolResult:
    capability = ACTION_CAPABILITY[arguments.action]
    try:
        check_capability(capability, getattr(context, "allowed_capabilities", ()))
    except PermissionDenied as exc:
        raise ToolError(str(exc)) from exc

    driver = getattr(context, "cua", None) or CuaDriver()
    try:
        if arguments.action == "move":
            if arguments.x is None or arguments.y is None:
                raise ToolError("move needs both x and y")
            x, y = driver.move(arguments.x, arguments.y)
            return ToolResult(text=f"moved the pointer to ({x}, {y})")
        if arguments.action == "click":
            x, y = driver.click(
                x=arguments.x,
                y=arguments.y,
                button=arguments.button,
                clicks=arguments.clicks,
            )
            return ToolResult(
                text=f"clicked {arguments.button} {arguments.clicks}x at ({x}, {y})"
            )
        if arguments.action == "scroll":
            amount = driver.scroll(arguments.amount)
            return ToolResult(text=f"scrolled {amount} steps")
        if arguments.action == "type":
            if not arguments.text:
                raise ToolError("type needs text")
            count = driver.type_text(arguments.text)
            return ToolResult(
                text=f"typed {count} characters",
                detail={"characters": count, "text": "[redacted]"},
            )
        if not arguments.key:
            raise ToolError("key needs a key name")
        key = driver.press(arguments.key)
        return ToolResult(text=f"pressed {key}")
    except CuaError as exc:
        raise ToolError(str(exc)) from exc
