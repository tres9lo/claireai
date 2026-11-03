"""Local speech input bounds, lazy installation and transcript handling."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from claireai.voice_input import (
    MAX_AUDIO_BYTES,
    LocalTranscriber,
    SpeechRecognitionUnavailable,
)


class StubModel:
    def __init__(self, duration: float = 2.0) -> None:
        self.duration = duration
        self.received = None
        self.options = None

    def transcribe(self, audio, **kwargs):  # noqa: ANN001
        self.received = audio.read()
        self.options = kwargs
        return iter([SimpleNamespace(text=" Hello locally "), SimpleNamespace(text="world.")]), SimpleNamespace(duration=self.duration)


def test_transcription_uses_memory_and_limits_generation_options(tmp_path: Path) -> None:
    model = StubModel()
    transcriber = LocalTranscriber(tmp_path / "cache", model_factory=lambda *a, **kw: model)

    result = transcriber.transcribe(b"webm bytes", content_type="audio/webm")

    assert result == "Hello locally world."
    assert model.received == b"webm bytes"
    assert model.options == {
        "beam_size": 1,
        "vad_filter": True,
        "condition_on_previous_text": False,
    }


def test_voice_model_download_is_explicitly_requested(tmp_path: Path) -> None:
    seen = {}

    def factory(model_id, **kwargs):  # noqa: ANN001
        seen.update(model_id=model_id, **kwargs)
        return StubModel()

    transcriber = LocalTranscriber(tmp_path / "cache", model_factory=factory)
    transcriber.install()

    assert seen["model_id"] == "Systran/faster-whisper-tiny"
    assert seen["local_files_only"] is False
    assert seen["device"] == "cpu" and seen["compute_type"] == "int8"


@pytest.mark.parametrize(
    ("audio", "content_type", "message"),
    [
        (b"audio", "text/plain", "unsupported audio format"),
        (b"", "audio/webm", "between 1 byte"),
        (b"x" * (MAX_AUDIO_BYTES + 1), "audio/webm", "between 1 byte"),
    ],
)
def test_transcription_rejects_invalid_audio(
    tmp_path: Path, audio: bytes, content_type: str, message: str
) -> None:
    transcriber = LocalTranscriber(tmp_path / "cache", model_factory=lambda *_a, **_kw: StubModel())
    with pytest.raises(ValueError, match=message):
        transcriber.transcribe(audio, content_type=content_type)


def test_transcription_rejects_long_recordings_before_decoding_segments(tmp_path: Path) -> None:
    model = StubModel(duration=21)
    transcriber = LocalTranscriber(tmp_path / "cache", model_factory=lambda *_a, **_kw: model)
    with pytest.raises(ValueError, match="exceeds 20 seconds"):
        transcriber.transcribe(b"audio", content_type="audio/wav")


def test_empty_speech_does_not_return_a_fabricated_transcript(tmp_path: Path) -> None:
    class Silence:
        def transcribe(self, *_args, **_kwargs):  # noqa: ANN002, ANN003
            return iter(()), SimpleNamespace(duration=1.0)

    transcriber = LocalTranscriber(tmp_path / "cache", model_factory=lambda *_a, **_kw: Silence())
    with pytest.raises(ValueError, match="no speech was recognized"):
        transcriber.transcribe(b"silence", content_type="audio/wav")


def test_missing_optional_runtime_has_actionable_error(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    transcriber = LocalTranscriber(tmp_path / "cache")
    monkeypatch.setattr("claireai.voice_input.importlib.util.find_spec", lambda _name: None)
    with pytest.raises(SpeechRecognitionUnavailable, match="voice extra"):
        transcriber.install()
