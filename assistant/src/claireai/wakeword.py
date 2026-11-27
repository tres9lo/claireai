"""Local wake-word detection with openWakeWord.

The engine consumes 16 kHz, mono, signed 16-bit PCM frames (80 ms = 1280
samples). It never writes audio to disk and never opens the microphone until
:class:`WakeWordListener` is explicitly started.

openWakeWord ships pretrained phrases (for example ``hey_jarvis`` or ``alexa``).
A custom "Hey Claire" model has to be trained and supplied as a model name or
path; this module does not silently download one.
"""

from __future__ import annotations

import array
import io
import math
import re
import shutil
import subprocess
import sys
import threading
import time
import types
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Type

SAMPLE_RATE = 16_000
FRAME_SAMPLES = 1_280  # 80 ms
FRAME_BYTES = FRAME_SAMPLES * 2  # int16
DEFAULT_COOLDOWN_SECONDS = 1.5


class WakeWordUnavailable(RuntimeError):
    """The optional runtime, model, or microphone is unavailable."""


@dataclass(frozen=True)
class Detection:
    phrase: str
    score: float
    at: float
    # Speech-to-text detection can carry the command from the same utterance
    # ("Hey Claire, open Downloads"), avoiding a second five-second recording.
    command: str = ""


def _model_class() -> Type[Any]:
    """Import the ONNX inference class without the optional training stack.

    openWakeWord's package ``__init__`` eagerly imports its custom-verifier
    trainer, which drags in scipy and scikit-learn. Those packages are only for
    training, not for inference, and scipy has no reliable wheel on some
    Python/network combinations. Stubbing just that module keeps the supported
    ONNX path usable while leaving training unavailable and clearly optional.
    """
    try:
        import openwakeword  # type: ignore[import-not-found]
        from openwakeword.model import Model  # type: ignore[import-not-found]

        if hasattr(openwakeword, "get_pretrained_model_paths"):
            return Model
        # A previous failed import can leave a partial package with just the
        # submodules cached. Fall through and rebuild it with the stub.
    except ModuleNotFoundError as exc:
        if exc.name not in {"scipy", "sklearn"}:
            raise
        stub = types.ModuleType("openwakeword.custom_verifier_model")

        def unavailable(*_args: Any, **_kwargs: Any) -> None:
            raise WakeWordUnavailable(
                "custom wake-word verifier training needs scipy and scikit-learn; "
                "inference is available without them"
            )

        stub.train_custom_verifier = unavailable  # type: ignore[attr-defined]
        # A failed package import can leave ``openwakeword`` and its submodules
        # cached but only partly initialized. Remove those partial modules while
        # keeping the training-only stub, then import the package afresh.
        for name in tuple(sys.modules):
            if name == "openwakeword" or name.startswith("openwakeword."):
                if name != "openwakeword.custom_verifier_model":
                    sys.modules.pop(name, None)
        sys.modules["openwakeword.custom_verifier_model"] = stub
        from openwakeword.model import Model  # type: ignore[import-not-found]

        return Model


class OpenWakeWordEngine:
    """One loaded openWakeWord model, with a cooldown between detections."""

    def __init__(
        self,
        model: str = "hey_jarvis",
        *,
        threshold: float = 0.5,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.model_name = str(model or "").strip()
        if not self.model_name:
            raise ValueError("a wake-word model name or path is required")
        if not 0 < float(threshold) <= 1:
            raise ValueError("wake-word threshold must be in (0, 1]")
        self.threshold = float(threshold)
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self.model_factory = model_factory
        self._model: Any = None
        self._lock = threading.Lock()
        self._last_at = 0.0

    @property
    def available(self) -> bool:
        if self.model_factory is not None:
            return True
        try:
            _model_class()
        except Exception:
            return False
        return True

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                if self.model_factory is not None:
                    self._model = self.model_factory(self.model_name)
                else:
                    Model = _model_class()

                    # Passing the name/path explicitly avoids the package's
                    # implicit "download everything" helper. ONNX is explicit
                    # because tflite-runtime has no Python 3.12 wheel.
                    self._model = Model(
                        wakeword_models=[self.model_name],
                        inference_framework="onnx",
                    )
            except Exception as exc:
                raise WakeWordUnavailable(
                    f"wake-word model {self.model_name!r} could not be loaded: "
                    f"{type(exc).__name__}: {exc}; install it with the openWakeWord "
                    "model downloader or pass a path to a trained model"
                ) from exc
        return self._model

    def process(self, frame: bytes, *, now: float | None = None) -> Detection | None:
        """Score one frame. Returns a detection only above threshold."""
        if len(frame) != FRAME_BYTES:
            raise WakeWordUnavailable(
                f"wake-word frames must be {FRAME_BYTES} bytes "
                f"({FRAME_SAMPLES} samples at {SAMPLE_RATE} Hz), got {len(frame)}"
            )
        try:
            import numpy as np
        except Exception as exc:  # pragma: no cover - openWakeWord installs numpy
            raise WakeWordUnavailable("numpy is required for wake-word detection") from exc
        model = self._load()
        samples = np.frombuffer(frame, dtype=np.int16)
        try:
            scores = model.predict(samples)
        except Exception as exc:
            raise WakeWordUnavailable(
                f"wake-word prediction failed ({type(exc).__name__})"
            ) from exc
        if not isinstance(scores, dict) or not scores:
            return None
        phrase, score = max(scores.items(), key=lambda item: float(item[1]))
        value = float(score)
        current = time.monotonic() if now is None else float(now)
        if value < self.threshold:
            return None
        if current - self._last_at < self.cooldown_seconds:
            return None
        self._last_at = current
        return Detection(phrase=str(phrase), score=value, at=current)


class WakeWordListener:
    """Capture microphone frames and invoke ``on_detection``.

    ``stop`` is a :class:`threading.Event`; setting it makes the loop exit at
    the next frame boundary. The stream is closed in ``finally`` and no audio
    buffer outlives the callback.
    """

    def __init__(self, engine: OpenWakeWordEngine, *, stream_factory: Any = None) -> None:
        self.engine = engine
        self.stream_factory = stream_factory

    @property
    def available(self) -> bool:
        if self.stream_factory is not None:
            return True
        try:
            import sounddevice as sd  # type: ignore[import-not-found]

            sd.query_devices()
        except Exception:
            # PortAudio is a system library; a pip install can succeed while
            # the shared object is absent. ALSA's arecord is a usable fallback
            # on Linux and avoids making libportaudio a hard requirement.
            return bool(shutil.which("arecord")) and self.engine.available
        return self.engine.available

    def _run_stream(
        self,
        *,
        stream_factory: Any,
        stop: threading.Event,
        on_detection: Callable[[Detection], None],
    ) -> None:
        try:
            stream = stream_factory(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=FRAME_SAMPLES,
            )
        except Exception as exc:
            raise WakeWordUnavailable(
                f"the microphone could not be opened ({type(exc).__name__})"
            ) from exc
        try:
            with stream:
                while not stop.is_set():
                    data, overflowed = stream.read(FRAME_SAMPLES)
                    if overflowed:
                        continue
                    detection = self.engine.process(bytes(data))
                    if detection is not None:
                        on_detection(detection)
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()

    def _run_arecord(
        self,
        *,
        stop: threading.Event,
        on_detection: Callable[[Detection], None],
    ) -> None:
        binary = shutil.which("arecord")
        if not binary:
            raise WakeWordUnavailable(
                "microphone capture needs sounddevice/PortAudio or ALSA arecord"
            )
        try:
            process = subprocess.Popen(
                [
                    binary, "-q", "-f", "S16_LE", "-r", str(SAMPLE_RATE),
                    "-c", "1", "-t", "raw",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            raise WakeWordUnavailable(
                f"the microphone could not be opened ({type(exc).__name__})"
            ) from exc
        try:
            assert process.stdout is not None
            while not stop.is_set():
                chunk = process.stdout.read(FRAME_BYTES)
                if len(chunk) < FRAME_BYTES:
                    break
                detection = self.engine.process(chunk)
                if detection is not None:
                    on_detection(detection)
        finally:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()

    def run(
        self,
        *,
        stop: threading.Event,
        on_detection: Callable[[Detection], None],
    ) -> None:
        if self.stream_factory is not None:
            self._run_stream(
                stream_factory=self.stream_factory,
                stop=stop,
                on_detection=on_detection,
            )
            return
        try:
            import sounddevice as sd  # type: ignore[import-not-found]

            sd.query_devices()
        except Exception:
            self._run_arecord(stop=stop, on_detection=on_detection)
            return
        self._run_stream(
            stream_factory=sd.RawInputStream,
            stop=stop,
            on_detection=on_detection,
        )


def _wav_from_pcm(pcm: bytes, *, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Wrap raw mono int16 PCM in a WAV container, entirely in memory."""
    if len(pcm) < FRAME_BYTES:
        raise WakeWordUnavailable("the microphone returned no audio for this turn")
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return output.getvalue()


def record_command(seconds: float = 5.0, *, binary: str | None = None) -> bytes:
    """Capture one bounded command and return WAV bytes without saving audio."""
    duration = max(1.0, min(float(seconds), 20.0))
    arecord = binary or shutil.which("arecord")
    if arecord:
        try:
            completed = subprocess.run(
                [
                    arecord, "-q", "-f", "S16_LE", "-r", str(SAMPLE_RATE),
                    "-c", "1", "-d", str(int(duration)), "-t", "raw",
                ],
                capture_output=True,
                timeout=duration + 10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise WakeWordUnavailable(
                f"the microphone could not record the command ({type(exc).__name__})"
            ) from exc
        if completed.returncode != 0:
            raise WakeWordUnavailable(
                f"the microphone could not record the command (arecord exited {completed.returncode})"
            )
        return _wav_from_pcm(completed.stdout)

    try:
        import sounddevice as sd  # type: ignore[import-not-found]

        frames = sd.rec(
            int(duration * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
        )
        sd.wait()
    except Exception as exc:
        raise WakeWordUnavailable(
            "command recording needs sounddevice/PortAudio or ALSA arecord"
        ) from exc
    return _wav_from_pcm(bytes(frames))


def _rms(frame: bytes) -> float:
    """Cheap 16-bit mono energy, used only to decide when speech starts/ends."""
    if len(frame) < 2:
        return 0.0
    samples = array.array("h")
    samples.frombytes(frame[: len(frame) // 2 * 2])
    if sys.byteorder != "little":
        samples.byteswap()
    return math.sqrt(sum(value * value for value in samples) / max(1, len(samples)))


def _normalise_speech(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def _wake_aliases(phrase: str) -> tuple[str, ...]:
    """Spoken variants a tiny ASR model commonly returns for the phrase."""
    base = _normalise_speech(phrase)
    aliases = {base} if base else set()
    if "claire" in base:
        aliases.update({"hey claire", "hey clair", "hey clare", "hey clear", "hi claire", "ok claire"})
    return tuple(sorted(aliases, key=len, reverse=True))


def _match_wake_phrase(phrase: str, text: str) -> tuple[str, str] | None:
    """Return (matched alias, command after it), or ``None``.

    Matching ignores case and punctuation, but the command is sliced from the
    original transcript so file and folder names keep their case.
    """
    for alias in _wake_aliases(phrase):
        words = alias.split()
        pattern = re.compile(
            r"\b" + r"[\W_]+".join(re.escape(word) for word in words) + r"\b",
            re.IGNORECASE,
        )
        match = pattern.search(text)
        if match is None:
            continue
        command = text[match.end() :].lstrip(" \t,.;:!?-")
        return alias, command.strip()
    return None


class SpeechWakeWordListener:
    """Recognise a spoken wake phrase locally, without a custom ONNX model.

    openWakeWord only knows the phrases it was trained on; the default
    ``hey_jarvis`` model cannot hear "Hey Claire" in the owner's voice. This
    listener uses a cheap energy gate and then the already-installed local
    faster-whisper model, so the owner's actual pronunciation is recognised.
    Audio is held in memory for one utterance only.
    """

    def __init__(
        self,
        *,
        phrase: str = "hey claire",
        transcribe: Callable[[bytes], str],
        stream_factory: Any = None,
        binary: str | None = None,
        calibration_seconds: float = 0.8,
        pre_roll_seconds: float = 0.45,
        min_speech_seconds: float = 0.35,
        silence_seconds: float = 0.7,
        max_speech_seconds: float = 6.0,
        threshold_floor: float = 900.0,
        threshold_scale: float = 1.8,
    ) -> None:
        self.phrase = str(phrase or "").strip()
        if not self.phrase:
            raise ValueError("a wake phrase is required for speech recognition")
        self.transcribe = transcribe
        self.stream_factory = stream_factory
        self.binary = binary
        self.calibration_frames = max(1, round(calibration_seconds * 1000 / 80))
        self.pre_roll_frames = max(1, round(pre_roll_seconds * 1000 / 80))
        self.min_speech_frames = max(1, round(min_speech_seconds * 1000 / 80))
        self.silence_frames = max(1, round(silence_seconds * 1000 / 80))
        self.max_speech_frames = max(self.min_speech_frames, round(max_speech_seconds * 1000 / 80))
        self.threshold_floor = float(threshold_floor)
        self.threshold_scale = float(threshold_scale)
        self._process: subprocess.Popen[bytes] | None = None
        self._process_lock = threading.Lock()

    @property
    def available(self) -> bool:
        if self.stream_factory is not None:
            return True
        if self.binary or shutil.which("arecord"):
            return True
        try:
            import sounddevice as sd  # type: ignore[import-not-found]

            sd.query_devices()
        except Exception:
            return False
        return True

    def stop(self) -> None:
        """Unblock a pending ``arecord`` read when Stop is pressed."""
        with self._process_lock:
            process = self._process
        if process is not None and process.poll() is None:
            process.terminate()

    def _process_reader(
        self,
        *,
        stop: threading.Event,
        on_detection: Callable[[Detection], None],
        read: Callable[[], bytes | None],
    ) -> None:
        noise = 0.0
        calibrated = 0
        buffered: list[bytes] = []
        voiced_run = 0
        silent_run = 0
        in_speech = False

        while not stop.is_set():
            frame = read()
            if not frame:
                break
            level = _rms(frame)
            if calibrated < self.calibration_frames:
                noise = level if calibrated == 0 else noise * 0.8 + level * 0.2
                calibrated += 1
                continue
            threshold = max(self.threshold_floor, noise * self.threshold_scale)
            voiced = level > threshold

            if not in_speech:
                # Keep the moment before the first loud frame. The first
                # syllable of "Hey" is often quieter than the rest, and
                # dropping it made the model hear only "open the notes".
                buffered.append(frame)
                if len(buffered) > self.pre_roll_frames:
                    buffered.pop(0)
                if voiced:
                    voiced_run += 1
                    if voiced_run >= 2:
                        in_speech = True
                        silent_run = 0
                else:
                    voiced_run = 0
                    noise = noise * 0.95 + level * 0.05
                continue

            buffered.append(frame)
            silent_run = 0 if voiced else silent_run + 1
            if len(buffered) < self.max_speech_frames and silent_run < self.silence_frames:
                continue

            speech_frames = max(0, len(buffered) - silent_run)
            pcm = b"".join(buffered[:speech_frames])
            buffered = []
            voiced_run = 0
            silent_run = 0
            in_speech = False
            if len(pcm) < self.min_speech_frames * FRAME_BYTES:
                continue
            try:
                text = self.transcribe(_wav_from_pcm(pcm)).strip()
            except Exception as exc:  # noqa: BLE001 - report through the service/UI
                raise WakeWordUnavailable(
                    f"local wake-phrase transcription failed ({type(exc).__name__})"
                ) from exc
            match = _match_wake_phrase(self.phrase, text)
            if match is None:
                continue
            alias, command = match
            on_detection(
                Detection(
                    phrase=alias or self.phrase,
                    score=1.0,
                    at=time.monotonic(),
                    command=command,
                )
            )
            return

    def _run_stream(
        self,
        *,
        stream_factory: Any,
        stop: threading.Event,
        on_detection: Callable[[Detection], None],
    ) -> None:
        try:
            stream = stream_factory(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=FRAME_SAMPLES,
            )
        except Exception as exc:
            raise WakeWordUnavailable(
                f"the microphone could not be opened ({type(exc).__name__})"
            ) from exc
        try:
            with stream:
                def read() -> bytes | None:
                    data, overflowed = stream.read(FRAME_SAMPLES)
                    return None if overflowed else bytes(data)

                self._process_reader(stop=stop, on_detection=on_detection, read=read)
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()

    def _run_arecord(
        self,
        *,
        stop: threading.Event,
        on_detection: Callable[[Detection], None],
    ) -> None:
        binary = self.binary or shutil.which("arecord")
        if not binary:
            raise WakeWordUnavailable(
                "microphone capture needs sounddevice/PortAudio or ALSA arecord"
            )
        try:
            process = subprocess.Popen(
                [
                    binary, "-q", "-f", "S16_LE", "-r", str(SAMPLE_RATE),
                    "-c", "1", "-t", "raw",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            raise WakeWordUnavailable(
                f"the microphone could not be opened ({type(exc).__name__})"
            ) from exc
        with self._process_lock:
            self._process = process
        try:
            assert process.stdout is not None

            def read() -> bytes | None:
                chunk = process.stdout.read(FRAME_BYTES)
                return chunk if len(chunk) == FRAME_BYTES else None

            self._process_reader(stop=stop, on_detection=on_detection, read=read)
        finally:
            with self._process_lock:
                if self._process is process:
                    self._process = None
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)

    def run(
        self,
        stop: threading.Event,
        on_detection: Callable[[Detection], None],
    ) -> None:
        if self.stream_factory is not None:
            self._run_stream(
                stream_factory=self.stream_factory,
                stop=stop,
                on_detection=on_detection,
            )
            return
        if self.binary or shutil.which("arecord"):
            self._run_arecord(stop=stop, on_detection=on_detection)
            return
        try:
            import sounddevice as sd  # type: ignore[import-not-found]

            sd.query_devices()
        except Exception as exc:
            raise WakeWordUnavailable(
                "microphone capture needs sounddevice/PortAudio or ALSA arecord"
            ) from exc
        self._run_stream(
            stream_factory=sd.RawInputStream,
            stop=stop,
            on_detection=on_detection,
        )


class WakeWordService:
    """Run detection, then capture and transcribe one command per activation.

    The listener is stopped while the command is recorded so one microphone
    stream owns the device at a time. Audio held in this class is transient: the
    raw capture becomes WAV bytes in memory and is never written to the data
    directory. The transcribed text is handed to ``on_command``; the service
    itself never grants a tool permission.
    """

    def __init__(
        self,
        *,
        model: str,
        threshold: float,
        data_dir: Path,
        engine_factory: Callable[[], "OpenWakeWordEngine"] | None = None,
        listener_factory: Callable[["OpenWakeWordEngine"], WakeWordListener] | None = None,
        recorder: Callable[[float], bytes] | None = None,
        transcriber: Callable[[bytes], str] | None = None,
        on_status: Callable[[str, dict[str, Any]], None] | None = None,
        record_seconds: float = 5.0,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
    ) -> None:
        self.model = model
        self.threshold = threshold
        self.data_dir = Path(data_dir)
        self._engine_factory = engine_factory
        self._listener_factory = listener_factory
        self._recorder = recorder or record_command
        self._transcriber = transcriber or self._default_transcriber
        self._on_status = on_status
        self._record_seconds = max(1.0, min(float(record_seconds), 20.0))
        self._cooldown = max(0.0, float(cooldown_seconds))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._on_command: Callable[[str], None] | None = None
        self._listener: WakeWordListener | None = None
        self._last: Detection | None = None
        self._error: str | None = None
        self._state = "stopped"

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def last_detection(self) -> Detection | None:
        return self._last

    @property
    def last_error(self) -> str | None:
        return self._error

    @property
    def state(self) -> str:
        return self._state

    def _status(self, state: str, **detail: Any) -> None:
        self._state = state
        if self._on_status is not None:
            try:
                self._on_status(state, detail)
            except Exception:  # a UI callback must not kill the microphone loop
                pass

    def _default_transcriber(self, audio: bytes) -> str:
        from claireai.voice_input import transcriber_for

        transcriber = transcriber_for(str(self.data_dir / "models" / "voice" / "asr"))
        if not transcriber.runtime_available:
            raise WakeWordUnavailable(
                "the wake word heard you, but speech recognition is not installed; "
                "install the voice extra and the local speech model"
            )
        if not transcriber.installed:
            raise WakeWordUnavailable(
                "the wake word heard you, but the local speech model is not installed; "
                "install it from Settings or the Voice page"
            )
        return transcriber.transcribe(
            audio, content_type="audio/wav", hotwords=self.model
        )

    def start(self, on_command: Callable[[str], None]) -> None:
        if self.running:
            return
        engine = (
            self._engine_factory()
            if self._engine_factory is not None
            else OpenWakeWordEngine(self.model, threshold=self.threshold)
        )
        listener = (
            self._listener_factory(engine)
            if self._listener_factory is not None
            else WakeWordListener(engine)
        )
        if not listener.available:
            raise WakeWordUnavailable(
                "wake-word runtime or microphone is unavailable; install the wakeword "
                "extra and an openWakeWord model"
            )
        self._listener = listener
        self._on_command = on_command
        self._stop.clear()
        self._error = None
        self._status("starting")
        self._thread = threading.Thread(
            target=self._run, name="claireai-wakeword", daemon=True
        )
        self._thread.start()

    def stop(self, *, timeout: float = 3.0) -> None:
        self._stop.set()
        listener_stop = getattr(self._listener, "stop", None)
        if callable(listener_stop):
            listener_stop()
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        self._state = "stopped"

    def _run(self) -> None:
        assert self._listener is not None
        self._status("listening")
        try:
            while not self._stop.is_set():
                detected: list[Detection] = []
                turn_stop = threading.Event()

                def heard(event: Detection) -> None:
                    detected.append(event)
                    turn_stop.set()

                try:
                    self._listener.run(stop=turn_stop, on_detection=heard)
                except WakeWordUnavailable as exc:
                    self._error = str(exc)
                    self._status("error", error=str(exc))
                    return
                if self._stop.is_set():
                    break
                if not detected:
                    # A stream that ended without a detection is retried rather
                    # than spinning; this also gives the device time to settle.
                    self._stop.wait(0.25)
                    continue

                event = detected[-1]
                self._last = event
                self._status(
                    "recording", phrase=event.phrase, score=event.score
                )
                try:
                    if event.command.strip():
                        # Speech-to-text heard the phrase and command in one
                        # breath; do not open the microphone for another turn.
                        text = event.command.strip()
                        self._status("transcribing", phrase=event.phrase)
                    else:
                        audio = self._recorder(self._record_seconds)
                        self._status("transcribing", phrase=event.phrase)
                        text = self._transcriber(audio).strip()
                    if not text:
                        raise WakeWordUnavailable("no speech was recognized after the wake word")
                    if self._on_command is not None:
                        self._on_command(text)
                    self._status("listening")
                except Exception as exc:  # noqa: BLE001 - reported in the UI
                    self._error = str(exc)
                    self._status("error", error=str(exc))
                self._stop.wait(self._cooldown)
        finally:
            self._status("stopped")


__all__ = [
    "DEFAULT_COOLDOWN_SECONDS",
    "Detection",
    "FRAME_BYTES",
    "FRAME_SAMPLES",
    "OpenWakeWordEngine",
    "SAMPLE_RATE",
    "SpeechWakeWordListener",
    "WakeWordListener",
    "WakeWordService",
    "WakeWordUnavailable",
    "record_command",
]
