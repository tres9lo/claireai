"""Optional offline speech output through a host-installed eSpeak executable."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path

MAX_SPOKEN_CHARS = 6000
MIN_PITCH, MAX_PITCH = 0, 99
MIN_RATE, MAX_RATE = 80, 450
SYNTH_TIMEOUT_SECONDS = 30
PLAYBACK_TIMEOUT_SECONDS = 90

# Only languages with a voice on the host are offered; both are checked before
# use. French read with an English voice is the defect this fixes.
SUPPORTED_LANGUAGES = {"en": "English", "fr": "French"}
FRENCH_MARKERS = frozenset(
    "le la les un une des du de et est je tu il elle nous vous ils elles pas que qui "
    "pour avec dans sur ce cette ces mais ou au aux etre avoir bonjour merci oui non "
    "tres aussi toujours maintenant aujourd hui tout tous faire peut plus bien".split()
)
ENGLISH_MARKERS = frozenset(
    "the and is are you i to of in that it for with on this be have not do what how "
    "please thanks hello yes no very also always now today my your we they can more "
    "all make get want like just about there when why where who".split()
)
_FRENCH_LETTERS = "àâçéèêëîïôûùüÿœæ"


def detect_language(text: str) -> str:
    """Pick ``en`` or ``fr`` for a piece of text.

    An accented letter settles it; otherwise the more frequent function words
    win, with English as the tie-break because the assistant's instructions and
    most of its answers are English. This is a voice choice, not a language
    model: it never changes what ClaireAI says.
    """
    lowered = text.lower()
    if any(letter in lowered for letter in _FRENCH_LETTERS):
        return "fr"
    words = re.findall(r"[a-z']+", lowered)
    if not words:
        return "en"
    french = sum(1 for word in words if word in FRENCH_MARKERS)
    english = sum(1 for word in words if word in ENGLISH_MARKERS)
    return "fr" if french > english else "en"


def for_speech(text: str) -> str:
    """Turn an answer into something worth hearing.

    Markdown is meant to be seen: fences, backticks, headings, emphasis, list
    bullets and link syntax all get read out as punctuation otherwise. Code and
    URLs are dropped rather than spelled out letter by letter.
    """
    cleaned = re.sub(r"```.*?```", " Code block omitted. ", text, flags=re.DOTALL)
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", cleaned)
    cleaned = re.sub(r"https?://\S+", " a link ", cleaned)
    cleaned = re.sub(r"^\s{0,3}#{1,6}\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"(\*\*|__|\*|_)(.*?)\1", r"\2", cleaned)
    cleaned = re.sub(r"^\s*[-*+]\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*>\s?", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"[|]{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def available_voices(engine: str | None = None) -> dict[str, str]:
    """Languages this host can actually speak, as ``code -> name``.

    A Piper voice counts on its own: with one installed, speech works even where
    eSpeak NG is absent.
    """
    found: dict[str, str] = {
        code: name for code, name in SUPPORTED_LANGUAGES.items() if piper_voice_for(code) is not None
    }
    executable = engine or shutil.which("espeak-ng") or shutil.which("espeak")
    if executable is None:
        return found
    for code, name in SUPPORTED_LANGUAGES.items():
        if code in found:
            continue
        try:
            listing = subprocess.run(
                [executable, f"--voices={code}"],
                capture_output=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if listing.returncode == 0 and listing.stdout.count(b"\n") > 1:
            found[code] = name
    return found


def piper_model_dirs() -> list[Path]:
    """Where a Piper neural voice may live, most specific first."""
    override = os.environ.get("CLAIREAI_PIPER_DIR")
    if override:
        # An explicit override replaces the search path rather than extending it,
        # so an operator (or a test) can pin exactly which voices are visible.
        return [Path(override).expanduser()]
    dirs: list[Path] = []
    dirs.append(Path.home() / ".local" / "share" / "ClaireAI" / "models" / "voice" / "piper")
    # <repo>/runtime-data is the development data directory. voice.py sits at
    # <repo>/assistant/src/claireai/voice.py, hence parents[3].
    dirs.append(Path(__file__).resolve().parents[3] / "runtime-data" / "models" / "voice" / "piper")
    return dirs


def piper_voice_for(language: str) -> Path | None:
    """A Piper ``.onnx`` voice for ``language``, or ``None`` if none is installed.

    Only a model whose own file name carries the language code is accepted, so a
    French sentence is never handed to an English voice. The ``medium`` voices
    are preferred because they are the smallest that still sound natural.
    """
    if language not in SUPPORTED_LANGUAGES:
        return None
    matches: list[Path] = []
    # Piper ships voices as ``<locale>-<speaker>-<quality>.onnx``, where the
    # locale starts the file name (``en_US-lessac-medium.onnx``). Accept a bare
    # language prefix as well as one embedded in the name.
    patterns = (f"{language}_*.onnx", f"{language}-*.onnx", f"*_{language}_*.onnx")
    for directory in piper_model_dirs():
        if not directory.is_dir():
            continue
        for pattern in patterns:
            for candidate in sorted(directory.glob(pattern)):
                if candidate.with_suffix(".onnx.json").is_file() and candidate not in matches:
                    matches.append(candidate)
    if not matches:
        return None
    matches.sort(key=lambda path: ("medium" not in path.name, path.name))
    return matches[0]


def synthesize_with_piper(spoken: str, model: Path, *, rate: int = 175) -> bytes:
    """Render ``spoken`` to WAV bytes with a local neural voice.

    Piper has no pitch control; the rate is expressed as ``length_scale`` so the
    existing rate slider still changes how fast it talks.
    """
    import io
    import wave

    from piper import PiperVoice, SynthesisConfig

    voice = PiperVoice.load(str(model))
    length_scale = max(0.5, min(2.5, 175.0 / max(1, rate)))
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        voice.synthesize_wav(spoken, wav, syn_config=SynthesisConfig(length_scale=length_scale))
    return buffer.getvalue()


class VoiceUnavailable(RuntimeError):
    """Speech synthesis or local audio playback is unavailable."""


class VoiceStopped(RuntimeError):
    """The owner stopped ClaireAI while it was speaking."""


def speak_text(
    text: str,
    *,
    pitch: int = 50,
    rate: int = 175,
    language: str | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> str:
    """Speak text locally, returning the engine name used.

    ``language`` is ``en`` or ``fr``; ``None`` classifies the text first so a
    French answer is never read with an English voice. Synthesized WAV bytes are
    piped directly to the host audio player; they are never written to disk.
    Text is sent on stdin and is not treated as a shell command or markup.
    """
    if type(pitch) is not int or not MIN_PITCH <= pitch <= MAX_PITCH:
        raise ValueError(f"pitch must be an integer from {MIN_PITCH} to {MAX_PITCH}")
    if type(rate) is not int or not MIN_RATE <= rate <= MAX_RATE:
        raise ValueError(f"rate must be an integer from {MIN_RATE} to {MAX_RATE}")
    if not isinstance(text, str):
        raise TypeError("speech text must be a string")
    if language is not None and language not in SUPPORTED_LANGUAGES:
        raise ValueError("language must be en, fr, or None")
    spoken = for_speech(text)[:MAX_SPOKEN_CHARS]
    if not spoken:
        raise ValueError("speech text must not be empty")
    chosen = language or detect_language(spoken)

    # A neural voice sounds like a person; eSpeak NG sounds like a machine. Use
    # Piper when a model for this language is installed, and keep eSpeak NG as
    # the always-available fallback rather than making speech conditional.
    audio: bytes | None = None
    engine_name = ""
    piper_model = piper_voice_for(chosen)
    if piper_model is not None:
        try:
            candidate = synthesize_with_piper(spoken, piper_model, rate=rate)
            if candidate.startswith(b"RIFF"):
                audio, engine_name = candidate, "Piper (neural)"
        except Exception:  # noqa: BLE001 - a broken model must fall back, not crash
            audio = None

    engine = shutil.which("espeak-ng") or shutil.which("espeak")
    if audio is None:
        if engine is None:
            raise VoiceUnavailable(
                "offline speech engine not found; install eSpeak NG or a Piper voice"
            )
        command = [
            engine, "--stdout", "--stdin",
            "-v", chosen,
            "-p", str(pitch), "-s", str(rate),
        ]
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
        audio = synthesized.stdout
        engine_name = "eSpeak NG" if engine.endswith("espeak-ng") else "eSpeak"
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
                winsound.PlaySound(audio, winsound.SND_MEMORY)
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
        return engine_name

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
            player_process.stdin.write(audio)
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
    return engine_name
