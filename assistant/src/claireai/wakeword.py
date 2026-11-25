"""Local wake-word detection with openWakeWord.

The engine consumes 16 kHz, mono, signed 16-bit PCM frames (80 ms = 1280
samples). It never writes audio to disk and never opens the microphone until
:class:`WakeWordListener` is explicitly started.

openWakeWord ships pretrained phrases (for example ``hey_jarvis`` or ``alexa``).
A custom "Hey Claire" model has to be trained and supplied as a model name or
path; this module does not silently download one.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import threading
import time
import types
from dataclasses import dataclass
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
