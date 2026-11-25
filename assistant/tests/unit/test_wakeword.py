"""Wake-word detection without openWakeWord or a microphone."""

from __future__ import annotations

import threading

import pytest

from claireai.wakeword import (
    FRAME_BYTES,
    OpenWakeWordEngine,
    WakeWordListener,
    WakeWordUnavailable,
)


class FakeModel:
    def __init__(self, score: float = 0.9) -> None:
        self.score = score
        self.frames = 0

    def predict(self, samples):  # noqa: ANN001
        self.frames += 1
        return {"hey_jarvis": self.score}


def test_a_frame_above_threshold_is_a_detection() -> None:
    model = FakeModel(0.9)
    engine = OpenWakeWordEngine(model_factory=lambda _name: model)
    detection = engine.process(b"\0" * FRAME_BYTES)
    assert detection is not None
    assert detection.phrase == "hey_jarvis"
    assert detection.score == 0.9


def test_a_frame_below_threshold_is_silent() -> None:
    engine = OpenWakeWordEngine(
        threshold=0.8, model_factory=lambda _name: FakeModel(0.4)
    )
    assert engine.process(b"\0" * FRAME_BYTES) is None


def test_cooldown_prevents_a_double_trigger() -> None:
    engine = OpenWakeWordEngine(
        cooldown_seconds=10, model_factory=lambda _name: FakeModel(0.9)
    )
    assert engine.process(b"\0" * FRAME_BYTES, now=100.0) is not None
    assert engine.process(b"\0" * FRAME_BYTES, now=101.0) is None
    assert engine.process(b"\0" * FRAME_BYTES, now=111.0) is not None


def test_the_wrong_frame_size_is_refused() -> None:
    engine = OpenWakeWordEngine(model_factory=lambda _name: FakeModel())
    with pytest.raises(WakeWordUnavailable, match="frames must be"):
        engine.process(b"\0" * 10)


class FakeStream:
    def __init__(self) -> None:
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):  # noqa: ANN002
        self.closed = True

    def read(self, _frames):
        return b"\0" * FRAME_BYTES, False

    def close(self):
        self.closed = True


def test_the_listener_stops_after_a_detection() -> None:
    stop = threading.Event()
    seen = []
    stream = FakeStream()
    engine = OpenWakeWordEngine(model_factory=lambda _name: FakeModel(0.9))
    listener = WakeWordListener(engine, stream_factory=lambda **_kwargs: stream)

    def detected(event):  # noqa: ANN001
        seen.append(event)
        stop.set()

    listener.run(stop=stop, on_detection=detected)

    assert len(seen) == 1
    assert stream.closed is True
