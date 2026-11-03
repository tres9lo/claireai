"""Optional offline speech output through a host-installed eSpeak executable."""

from __future__ import annotations

import shutil
import subprocess
import os
import threading
import time
from collections.abc import Callable

MAX_SPOKEN_CHARS = 6000
MIN_PITCH, MAX_PITCH = 0, 99
MIN_RATE, MAX_RATE = 80, 450
SYNTH_TIMEOUT_SECONDS = 30
PLAYBACK_TIMEOUT_SECONDS = 90


class VoiceUnavailable(RuntimeError):
    """Speech synthesis or local audio playback is unavailable."""


class VoiceStopped(RuntimeError):
    """The owner stopped ClaireAI while it was speaking."""


def speak_text(
    text: str,
    *,
    pitch: int = 50,
    rate: int = 175,
    should_stop: Callable[[], bool] | None = None,
) -> str:
    """Speak text locally, returning the engine name used.

    Synthesized WAV bytes are piped directly to the host audio player; they are
    never written to disk. Text is sent on stdin and is not treated as a shell
    command or markup.
    """
    if type(pitch) is not int or not MIN_PITCH <= pitch <= MAX_PITCH:
        raise ValueError(f"pitch must be an integer from {MIN_PITCH} to {MAX_PITCH}")
    if type(rate) is not int or not MIN_RATE <= rate <= MAX_RATE:
        raise ValueError(f"rate must be an integer from {MIN_RATE} to {MAX_RATE}")
    if not isinstance(text, str):
        raise TypeError("speech text must be a string")
    spoken = text.strip()[:MAX_SPOKEN_CHARS]
    if not spoken:
        raise ValueError("speech text must not be empty")

    engine = shutil.which("espeak-ng") or shutil.which("espeak")
    if engine is None:
        raise VoiceUnavailable(
            "offline speech engine not found; install eSpeak NG (plus a local audio player)"
        )

    command = [engine, "--stdout", "--stdin", "-p", str(pitch), "-s", str(rate)]
    try:
        synthesized = subprocess.run(
            command,
            input=spoken.encode("utf-8"),
            capture_output=True,
            timeout=SYNTH_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise VoiceUnavailable(f"offline speech synthesis failed: {type(exc).__name__}") from exc
    if synthesized.returncode != 0 or not synthesized.stdout.startswith(b"RIFF"):
        raise VoiceUnavailable("offline speech engine returned no valid WAV audio")
    if should_stop is not None and should_stop():
        raise VoiceStopped("ClaireAI was stopped while preparing speech output")

    if os.name == "nt":
        try:
            import winsound
        except ImportError as exc:  # pragma: no cover - Windows-only
            raise VoiceUnavailable("Windows audio playback is unavailable") from exc
        finished = threading.Event()
        playback_errors: list[Exception] = []

        def play_windows_audio() -> None:  # pragma: no cover - Windows-only
            try:
                winsound.PlaySound(synthesized.stdout, winsound.SND_MEMORY)
            except Exception as exc:
                playback_errors.append(exc)
            finally:
                finished.set()

        threading.Thread(target=play_windows_audio, name="claireai-voice-audio", daemon=True).start()
        deadline = time.monotonic() + PLAYBACK_TIMEOUT_SECONDS
        while not finished.wait(0.05):
            if should_stop is not None and should_stop():
                winsound.PlaySound(None, winsound.SND_PURGE)
                raise VoiceStopped("ClaireAI was stopped while speaking")
            if time.monotonic() >= deadline:
                winsound.PlaySound(None, winsound.SND_PURGE)
                raise VoiceUnavailable("local audio playback exceeded its time limit")
        if playback_errors:
            raise VoiceUnavailable("Windows could not play the generated speech") from playback_errors[0]
        return "eSpeak NG" if engine.endswith("espeak-ng") else "eSpeak"

    players = ("paplay", "pw-play", "aplay")
    player = next((path for name in players if (path := shutil.which(name))), None)
    if player is None:
        raise VoiceUnavailable("no local audio player found (paplay, pw-play, or aplay)")
    player_args = [player, "-"] if player.endswith(("paplay", "pw-play")) else [player, "-q", "-"]
    try:
        player_process = subprocess.Popen(
            player_args,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        raise VoiceUnavailable(f"local audio playback failed: {type(exc).__name__}") from exc

    def feed_audio() -> None:
        try:
            assert player_process.stdin is not None
            player_process.stdin.write(synthesized.stdout)
            player_process.stdin.close()
        except (BrokenPipeError, OSError):
            return

    writer = threading.Thread(target=feed_audio, name="claireai-voice-audio", daemon=True)
    writer.start()
    deadline = time.monotonic() + PLAYBACK_TIMEOUT_SECONDS
    while player_process.poll() is None:
        if should_stop is not None and should_stop():
            player_process.terminate()
            try:
                player_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                player_process.kill()
                player_process.wait()
            raise VoiceStopped("ClaireAI was stopped while speaking")
        if time.monotonic() >= deadline:
            player_process.kill()
            player_process.wait()
            raise VoiceUnavailable("local audio playback exceeded its time limit")
        time.sleep(0.05)
    writer.join(timeout=1)
    if player_process.returncode != 0:
        raise VoiceUnavailable("local audio player could not play the generated speech")
    return "eSpeak NG" if engine.endswith("espeak-ng") else "eSpeak"
