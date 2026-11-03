"""Local TTS subprocess handling and input bounds."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from claireai.voice import VoiceUnavailable, speak_text


def test_voice_pipes_text_to_local_engine_and_audio_to_player(monkeypatch) -> None:
    # This case covers the eSpeak NG fallback, so no neural voice is offered.
    monkeypatch.setattr("claireai.voice.piper_voice_for", lambda _language: None)
    programs = {"espeak-ng": "/opt/bin/espeak-ng", "paplay": "/usr/bin/paplay"}
    monkeypatch.setattr("claireai.voice.shutil.which", programs.get)
    calls = []

    def run(command, **kwargs):  # noqa: ANN001
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=b"RIFFfake wav", stderr=b"")

    class FakeStdin:
        def __init__(self) -> None:
            self.data = bytearray()

        def write(self, data: bytes) -> None:
            self.data.extend(data)

        def close(self) -> None:
            pass

    class FakePlayer:
        def __init__(self, command, **kwargs) -> None:  # noqa: ANN001
            calls.append((command, kwargs))
            self.stdin = FakeStdin()
            self.returncode = 0

        def poll(self) -> int:
            return 0

    monkeypatch.setattr("claireai.voice.subprocess.Popen", FakePlayer)

    monkeypatch.setattr("claireai.voice.subprocess.run", run)

    assert speak_text("Hello Ange", pitch=63, rate=190) == "eSpeak NG"
    synth_command, synth_options = calls[0]
    # English text is spoken with the English voice, never the default voice of
    # whatever locale the engine happens to start in.
    assert synth_command == [
        "/opt/bin/espeak-ng", "--stdout", "--stdin", "-v", "en", "-p", "63", "-s", "190"
    ]
    assert synth_options["input"] == b"Hello Ange"
    assert calls[1][0] == ["/usr/bin/paplay", "-"]
    assert calls[1][1]["stdin"] is not None


def test_a_neural_voice_is_preferred_when_one_is_installed(monkeypatch) -> None:
    """Piper replaces eSpeak NG silently when a model for the language exists."""
    from pathlib import Path

    spoken = []
    monkeypatch.setattr("claireai.voice.piper_voice_for", lambda language: Path(f"/models/x_{language}_v.onnx"))

    def synth(text, model, *, rate=175):  # noqa: ANN001
        spoken.append((text, model, rate))
        return b"RIFFneural"

    monkeypatch.setattr("claireai.voice.synthesize_with_piper", synth)
    monkeypatch.setattr("claireai.voice.shutil.which", lambda name: "/usr/bin/paplay" if name == "paplay" else None)

    class FakePlayer:
        def __init__(self, command, **kwargs) -> None:  # noqa: ANN001
            self.stdin = SimpleNamespace(write=lambda data: None, close=lambda: None)
            self.returncode = 0

        def poll(self) -> int:
            return 0

    monkeypatch.setattr("claireai.voice.subprocess.Popen", FakePlayer)

    assert speak_text("Hello", rate=190) == "Piper (neural)"
    assert spoken == [("Hello", Path("/models/x_en_v.onnx"), 190)]


def test_a_broken_neural_voice_falls_back_instead_of_failing(monkeypatch) -> None:
    from pathlib import Path

    def explode(*_args, **_kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("model file is corrupt")

    monkeypatch.setattr("claireai.voice.piper_voice_for", lambda language: Path("/models/broken_en.onnx"))
    monkeypatch.setattr("claireai.voice.synthesize_with_piper", explode)
    programs = {"espeak-ng": "/opt/bin/espeak-ng", "paplay": "/usr/bin/paplay"}
    monkeypatch.setattr("claireai.voice.shutil.which", programs.get)
    monkeypatch.setattr(
        "claireai.voice.subprocess.run",
        lambda command, **kwargs: SimpleNamespace(returncode=0, stdout=b"RIFFfallback", stderr=b""),
    )

    class FakePlayer:
        def __init__(self, command, **kwargs) -> None:  # noqa: ANN001
            self.stdin = SimpleNamespace(write=lambda data: None, close=lambda: None)
            self.returncode = 0

        def poll(self) -> int:
            return 0

    monkeypatch.setattr("claireai.voice.subprocess.Popen", FakePlayer)

    assert speak_text("Hello") == "eSpeak NG"


def test_a_voice_model_must_match_the_language(tmp_path, monkeypatch) -> None:
    from claireai import voice

    monkeypatch.setenv("CLAIREAI_PIPER_DIR", str(tmp_path))
    (tmp_path / "en_US-lessac-medium.onnx").write_bytes(b"x")
    (tmp_path / "en_US-lessac-medium.onnx.json").write_text("{}", encoding="utf-8")

    # The model exists, but it is English: French must not borrow it.
    assert voice.piper_voice_for("en") == tmp_path / "en_US-lessac-medium.onnx"
    assert voice.piper_voice_for("fr") is None


def test_a_model_without_its_config_is_not_used(tmp_path, monkeypatch) -> None:
    from claireai import voice

    monkeypatch.setenv("CLAIREAI_PIPER_DIR", str(tmp_path))
    (tmp_path / "fr_FR-siwis-medium.onnx").write_bytes(b"x")

    assert voice.piper_voice_for("fr") is None


def test_owner_stop_terminates_audio_playback(monkeypatch) -> None:
    from claireai.voice import VoiceStopped
    monkeypatch.setattr("claireai.voice.piper_voice_for", lambda _language: None)

    monkeypatch.setattr(
        "claireai.voice.shutil.which",
        lambda name: {"espeak-ng": "/opt/bin/espeak-ng", "paplay": "/usr/bin/paplay"}.get(name),
    )
    monkeypatch.setattr(
        "claireai.voice.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout=b"RIFFfake wav", stderr=b""),
    )

    class FakeStdin:
        def write(self, _data: bytes) -> None:
            pass

        def close(self) -> None:
            pass

    class FakePlayer:
        def __init__(self, *_args, **_kwargs) -> None:  # noqa: ANN002, ANN003
            self.stdin = FakeStdin()
            self.returncode = None

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            self.returncode = -15

        def wait(self, timeout=None):  # noqa: ANN001
            return self.returncode

    monkeypatch.setattr("claireai.voice.subprocess.Popen", FakePlayer)
    stop_checks = iter((False, True))
    with pytest.raises(VoiceStopped, match="stopped while speaking"):
        speak_text("hello", should_stop=lambda: next(stop_checks))


@pytest.mark.parametrize("pitch,rate", [(-1, 175), (100, 175), (50, 79), (50, 451)])
def test_voice_rejects_out_of_range_tuning(pitch: int, rate: int) -> None:
    with pytest.raises(ValueError):
        speak_text("hello", pitch=pitch, rate=rate)


def test_voice_fails_clearly_if_engine_is_missing(monkeypatch) -> None:
    # No eSpeak NG and no neural voice: there is genuinely nothing to speak with.
    monkeypatch.setattr("claireai.voice.piper_voice_for", lambda _language: None)
    monkeypatch.setattr("claireai.voice.shutil.which", lambda _name: None)
    with pytest.raises(VoiceUnavailable, match="install eSpeak NG"):
        speak_text("hello")
