"""The CUA driver without a real desktop or a real pointer."""

from __future__ import annotations

import pytest

from claireai.cua import CuaDriver, CuaError
from claireai.permissions import Sandbox
from claireai.tools.base import ToolContext, ToolError
from claireai.tools.computer import ComputerControlArgs, computer_control


class FakeGui:
    FAILSAFE = False
    PAUSE = 0.0

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []
        self.x, self.y = 10, 20

    def position(self):
        class Point:
            x, y = 10, 20

        return Point()

    def moveTo(self, x, y, *, duration):  # noqa: N802
        self.calls.append(("moveTo", (x, y), {"duration": duration}))
        self.x, self.y = x, y

    def click(self, *, x, y, button, clicks):  # noqa: N802
        self.calls.append(("click", (x, y), {"button": button, "clicks": clicks}))

    def write(self, text, *, interval):
        self.calls.append(("write", (text,), {"interval": interval}))

    def press(self, key):
        self.calls.append(("press", (key,), {}))

    def hotkey(self, *keys):
        self.calls.append(("hotkey", keys, {}))

    def scroll(self, amount):
        self.calls.append(("scroll", (amount,), {}))


def context(*capabilities: str, gui: FakeGui | None = None) -> ToolContext:
    return ToolContext(
        sandbox=Sandbox(roots=()),
        allowed_capabilities=tuple(capabilities),
        cua=CuaDriver(gui or FakeGui()),
    )


def test_move_click_and_scroll_use_the_driver() -> None:
    gui = FakeGui()
    driver = CuaDriver(gui)
    assert driver.move(100, 200) == (100, 200)
    driver.click(x=100, y=200, clicks=2, button="right")
    driver.scroll(-4)
    assert [call[0] for call in gui.calls] == ["moveTo", "click", "scroll"]


def test_typing_presses_enter_between_lines() -> None:
    gui = FakeGui()
    count = CuaDriver(gui).type_text("hello\nworld")
    assert count == 11
    assert [call[0] for call in gui.calls] == ["write", "press", "write"]


def test_a_chord_is_sent_as_a_hotkey() -> None:
    gui = FakeGui()
    assert CuaDriver(gui).press("ctrl+s") == "ctrl+s"
    assert gui.calls == [("hotkey", ("ctrl", "s"), {})]


def test_bad_coordinates_are_refused() -> None:
    with pytest.raises(CuaError, match="x must be"):
        CuaDriver(FakeGui()).move(-1, 2)


def test_tool_refuses_an_action_outside_the_enabled_capability() -> None:
    with pytest.raises(ToolError, match="keyboard"):
        computer_control(
            context("cursor"),
            ComputerControlArgs(action="type", text="secret"),
        )


def test_tool_runs_a_click_with_cursor_permission() -> None:
    result = computer_control(
        context("cursor"),
        ComputerControlArgs(action="click", x=50, y=60),
    )
    assert "clicked" in result.text


def test_typed_text_is_not_in_the_tool_result() -> None:
    result = computer_control(
        context("keyboard"),
        ComputerControlArgs(action="type", text="my password"),
    )
    assert "my password" not in result.text
    assert result.detail["text"] == "[redacted]"
