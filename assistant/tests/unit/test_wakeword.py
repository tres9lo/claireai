"""Wake-word detection without openWakeWord or a microphone."""

from __future__ import annotations

import threading

import pytest

from claireai.wakeword import (
    FRAME_BYTES,
    Detection,
    OpenWakeWordEngine,
    SpeechWakeWordListener,
    WakeWordListener,
    WakeWordService,
    WakeWordUnavailable,
    _match_wake_phrase,
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


class OneShotListener:
    available = True

    def __init__(self, seen):  # noqa: ANN001
        self.seen = seen
        self.fired = False

    def run(self, *, stop, on_detection):  # noqa: ANN001
        if self.fired:
            stop.wait(1.0)
            return
        self.fired = True
        on_detection(Detection(phrase="hey_jarvis", score=0.99, at=1.0))
        stop.set()


def test_wake_phrase_matching_keeps_command_case_and_punctuation() -> None:
    assert _match_wake_phrase("hey claire", "Hey, Claire, open Notes.txt.") == (
        "hey claire",
        "open Notes.txt.",
    )
    assert _match_wake_phrase("hey claire", "hey clear what time is it") == (
        "hey clear",
        "what time is it",
    )
    assert _match_wake_phrase("hey claire", "open the notes") is None


class SpeechStream:
    def __init__(self, frames):  # noqa: ANN001
        self.frames = list(frames)
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):  # noqa: ANN002
        self.closed = True

    def read(self, _frames):
        return (self.frames.pop(0) if self.frames else b"\0" * FRAME_BYTES), False

    def close(self):
        self.closed = True


def test_speech_wake_phrase_recognises_the_command_in_one_breath() -> None:
    loud = b"\x00\x40" * (FRAME_BYTES // 2)
    quiet = b"\0" * FRAME_BYTES
    frames = [quiet, quiet] + [loud] * 5 + [quiet] * 3
    heard = []
    stop = threading.Event()

    def transcribe(audio: bytes) -> str:
        assert audio.startswith(b"RIFF")
        return "Hey Claire, open Notes.txt."

    listener = SpeechWakeWordListener(
        phrase="hey claire",
        transcribe=transcribe,
        stream_factory=lambda **_kwargs: SpeechStream(frames),
        calibration_seconds=0.16,
        min_speech_seconds=0.08,
        silence_seconds=0.16,
    )

    def detected(event):  # noqa: ANN001
        heard.append(event)
        stop.set()

    listener.run(stop=stop, on_detection=detected)

    assert len(heard) == 1
    assert heard[0].phrase == "hey claire"
    assert heard[0].command == "open Notes.txt."


def test_stop_unblocks_a_listener_that_is_waiting_for_the_microphone(tmp_path) -> None:
    listeners = []

    class BlockingListener:
        available = True

        def __init__(self) -> None:
            self.stopped = False
            self.stop_event = None

        def run(self, *, stop, on_detection):  # noqa: ANN001
            self.stop_event = stop
            stop.wait(5)

        def stop(self):
            self.stopped = True
            if self.stop_event is not None:
                self.stop_event.set()

    def factory(_engine):  # noqa: ANN001
        listener = BlockingListener()
        listeners.append(listener)
        return listener

    service = WakeWordService(
        model="hey_jarvis",
        threshold=0.5,
        data_dir=tmp_path,
        engine_factory=lambda: object(),
        listener_factory=factory,
        on_status=lambda _state, _detail: None,
    )
    service.start(lambda _text: None)
    assert service.running
    service.stop(timeout=2)

    assert listeners[0].stopped is True
    assert service.running is False


def test_service_uses_a_command_heard_with_the_phrase_without_recording_again(tmp_path) -> None:
    commands: list[str] = []
    recorded: list[float] = []
    done = threading.Event()

    class CommandListener:
        available = True

        def __init__(self) -> None:
            self.fired = False

        def run(self, *, stop, on_detection):  # noqa: ANN001
            if self.fired:
                stop.wait(1.0)
                return
            self.fired = True
            on_detection(
                Detection(
                    phrase="hey claire",
                    score=1.0,
                    at=1.0,
                    command="open the notes",
                )
            )
            stop.set()

    service = WakeWordService(
        model="hey claire",
        threshold=0.5,
        data_dir=tmp_path,
        engine_factory=lambda: object(),
        listener_factory=lambda _engine: CommandListener(),
        recorder=lambda seconds: recorded.append(seconds) or b"audio",
        transcriber=lambda _audio: "wrong second transcription",
        on_status=lambda _state, _detail: None,
        cooldown_seconds=0,
    )

    def command(text: str) -> None:
        commands.append(text)
        done.set()

    service.start(command)
    assert done.wait(2)
    service.stop()

    assert commands == ["open the notes"]
    assert recorded == []


def test_service_records_and_transcribes_one_command(tmp_path) -> None:
    import threading

    done = threading.Event()
    commands: list[str] = []

    def listener_factory(_engine):  # noqa: ANN001
        return OneShotListener(commands)

    service = WakeWordService(
        model="hey_jarvis",
        threshold=0.5,
        data_dir=tmp_path,
        engine_factory=lambda: object(),
        listener_factory=listener_factory,
        recorder=lambda _seconds: b"audio",
        transcriber=lambda _audio: "open the notes folder",
        on_status=lambda _state, _detail: None,
        cooldown_seconds=0,
    )

    def command(text: str) -> None:
        commands.append(text)
        done.set()

    service.start(command)
    assert done.wait(2)
    service.stop()

    assert commands == ["open the notes folder"]
    assert service.last_detection is not None
    assert service.last_detection.score == 0.99
