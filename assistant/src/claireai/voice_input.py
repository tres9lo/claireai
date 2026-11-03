"""Optional, local push-to-talk transcription using a separately licensed model."""

from __future__ import annotations

import functools
import importlib.util
import io
import threading
from pathlib import Path
from typing import Any

MODEL_ID = "Systran/faster-whisper-tiny"
MAX_AUDIO_BYTES = 10 * 1024 * 1024
MAX_AUDIO_SECONDS = 20
SUPPORTED_AUDIO_TYPES = frozenset(
    {"audio/webm", "audio/ogg", "audio/mp4", "audio/wav", "audio/mpeg"}
)


class SpeechRecognitionUnavailable(RuntimeError):
    """The optional speech runtime or model is unavailable."""


class LocalTranscriber:
    """Lazy CPU transcription; audio is decoded from memory and never saved."""

    def __init__(self, model_cache: Path, *, model_factory: Any = None) -> None:
        self.model_cache = Path(model_cache)
        self.model_factory = model_factory
        self._model: Any = None
        self._lock = threading.Lock()

    def _load(self, *, allow_download: bool) -> Any:
        if self._model is not None:
            return self._model
        try:
            if self.model_factory is None:
                from faster_whisper import WhisperModel
                model_factory = WhisperModel
            else:
                model_factory = self.model_factory
        except ImportError as exc:
            raise SpeechRecognitionUnavailable(
                "speech recognition is not installed; install ClaireAI's optional voice extra"
            ) from exc
        try:
            self.model_cache.mkdir(parents=True, exist_ok=True)
            self._model = model_factory(
                MODEL_ID,
                device="cpu",
                compute_type="int8",
                cpu_threads=2,
                num_workers=1,
                download_root=str(self.model_cache),
                local_files_only=not allow_download,
            )
        except Exception as exc:
            self._model = None
            if not allow_download:
                raise SpeechRecognitionUnavailable(
                    "local speech model is not installed; use Install speech model in the UI"
                ) from exc
            raise SpeechRecognitionUnavailable(
                f"local speech model download or load failed ({type(exc).__name__})"
            ) from exc
        return self._model

    def install(self) -> None:
        """Explicitly download/load the selected model from its public repo."""
        with self._lock:
            self._load(allow_download=True)

    @property
    def installed(self) -> bool:
        model_root = self.model_cache / "models--Systran--faster-whisper-tiny" / "snapshots"
        return any((snapshot / "model.bin").is_file() for snapshot in model_root.glob("*"))

    @property
    def runtime_available(self) -> bool:
        return self.model_factory is not None or importlib.util.find_spec("faster_whisper") is not None

    def transcribe(self, audio: bytes, *, content_type: str) -> str:
        if content_type not in SUPPORTED_AUDIO_TYPES:
            raise ValueError("unsupported audio format from the browser recorder")
        if not audio or len(audio) > MAX_AUDIO_BYTES:
            raise ValueError(f"audio must be between 1 byte and {MAX_AUDIO_BYTES} bytes")
        with self._lock:
            model = self._load(allow_download=False)
            try:
                segments, info = model.transcribe(
                    io.BytesIO(audio),
                    beam_size=1,
                    vad_filter=True,
                    condition_on_previous_text=False,
                )
                if float(info.duration) > MAX_AUDIO_SECONDS:
                    raise ValueError(f"recording exceeds {MAX_AUDIO_SECONDS} seconds")
                text = " ".join(segment.text.strip() for segment in segments).strip()
            except ValueError:
                raise
            except Exception as exc:
                raise SpeechRecognitionUnavailable(
                    f"local transcription failed ({type(exc).__name__})"
                ) from exc
        if not text:
            raise ValueError("no speech was recognized; try again or type your message")
        return text[:6000]


@functools.lru_cache(maxsize=2)
def transcriber_for(model_cache: str) -> LocalTranscriber:
    """Keep one lazy model instance per local data directory."""
    return LocalTranscriber(Path(model_cache))
