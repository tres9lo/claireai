"""The design tokens and the animation clock, tested without a display."""

from __future__ import annotations

import pytest

from claireai.desktop import theme


# -- colour maths ----------------------------------------------------------


def test_colours_round_trip_through_hex() -> None:
    for value in ("#000000", "#5ee0ff", "#a78bfa", "#ffffff"):
        assert theme.rgb_to_hex(theme.hex_to_rgb(value)) == value


def test_a_short_or_malformed_colour_is_refused() -> None:
    with pytest.raises(ValueError):
        theme.hex_to_rgb("#fff")
    with pytest.raises(ValueError):
        theme.hex_to_rgb("transparent")


def test_lerp_hits_both_ends_exactly() -> None:
    assert theme.lerp_color("#000000", "#ffffff", 0) == "#000000"
    assert theme.lerp_color("#000000", "#ffffff", 1) == "#ffffff"
    assert theme.lerp_color("#000000", "#ffffff", 0.5) == "#808080"


def test_lerp_clamps_outside_the_unit_interval() -> None:
    assert theme.lerp_color("#123456", "#abcdef", -3) == "#123456"
    assert theme.lerp_color("#123456", "#abcdef", 4) == "#abcdef"


def test_every_easing_starts_at_zero_and_ends_at_one() -> None:
    for name, curve in theme.EASINGS.items():
        assert curve(0) == pytest.approx(0, abs=1e-9), name
        assert curve(1) == pytest.approx(1, abs=1e-9), name


def test_easing_clamps_input_rather_than_extrapolating() -> None:
    for curve in theme.EASINGS.values():
        assert curve(-5) == pytest.approx(0, abs=1e-9)
        assert curve(9) == pytest.approx(1, abs=1e-9)


def test_ease_out_cubic_never_moves_backwards() -> None:
    previous = -1.0
    for step in range(51):
        value = theme.ease_out_cubic(step / 50)
        assert value >= previous
        previous = value


# -- animator --------------------------------------------------------------


class FakeClock:
    """A root whose clock we control, so tweens finish deterministically."""

    def __init__(self) -> None:
        self.now = 0.0
        self.pending: dict[str, callable] = {}
        self.cancelled: list[str] = []
        self._ids = 0

    def after(self, _milliseconds: int, callback) -> str:  # noqa: ANN001
        self._ids += 1
        token = f"job{self._ids}"
        self.pending[token] = callback
        return token

    def after_cancel(self, token: str) -> None:
        self.cancelled.append(token)
        self.pending.pop(token, None)

    def advance(self, seconds: float) -> None:
        """Move the clock, then run whatever frame is waiting."""
        self.now += seconds
        while self.pending:
            token, callback = next(iter(self.pending.items()))
            self.pending.pop(token)
            callback()


def test_a_tween_drives_the_value_to_its_end_and_stops(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock()
    monkeypatch.setattr(theme.time, "monotonic", lambda: clock.now)
    animator = theme.Animator(clock)

    seen: list[float] = []
    animator.tween("k", 0.0, 100.0, 100, seen.append, easing="linear")

    assert seen[0] == 0.0, "the first frame is applied immediately, with no flicker"
    clock.advance(0.2)  # well past the 100 ms duration
    assert seen[-1] == pytest.approx(100.0)
    assert animator._jobs == {}, "a finished tween leaves no timer behind"


def test_starting_a_tween_on_the_same_key_replaces_the_old_one(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock()
    monkeypatch.setattr(theme.time, "monotonic", lambda: clock.now)
    animator = theme.Animator(clock)

    animator.tween("nav", 0.0, 1.0, 200, lambda value: None)
    first = set(animator._jobs)
    animator.tween("nav", 0.0, 1.0, 200, lambda value: None)

    assert first, "the first tween registered a timer"
    assert len(animator._jobs) == 1, "one timer per key, so animations cannot fight"


def test_cancelling_and_stopping_leave_no_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock()
    monkeypatch.setattr(theme.time, "monotonic", lambda: clock.now)
    animator = theme.Animator(clock)

    animator.tween("a", 0.0, 1.0, 500, lambda value: None)
    animator.tween("b", 0.0, 1.0, 500, lambda value: None)
    animator.cancel("a")
    assert "a" not in [key for key in animator._jobs]

    animator.stop_all()
    assert animator._jobs == {}


def test_a_widget_that_disappears_mid_tween_does_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock()
    monkeypatch.setattr(theme.time, "monotonic", lambda: clock.now)
    animator = theme.Animator(clock)

    def vanish(_value: float) -> None:
        raise RuntimeError("window destroyed")

    animator.tween("gone", 0.0, 1.0, 100, vanish, easing="linear")
    clock.advance(0.5)  # must not propagate the widget's error
