"""Validated runtime configuration.

Configuration is validated once at startup and fails loudly: a partially
configured assistant that silently points at the wrong model or the wrong data
directory is worse than one that refuses to start.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from platformdirs import user_data_path
from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .endpoint import DEFAULT_BASE_URLS, check_endpoint

APP_NAME = "ClaireAI"

# The runtime is Ollama, not a checkpoint trained in this repository. These are
# the model tags `ollama pull` understands; the coder tag is used for requests
# that are clearly about writing or debugging code.
DEFAULT_OLLAMA_MODEL = "qwen3:1.7b"
DEFAULT_OLLAMA_CODING_MODEL = "qwen2.5-coder:1.5b"
# A separate small embedding model keeps semantic memory local. It is only
# pulled when the owner uses memory; the chat models are unchanged.
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_WAKEWORD_MODEL = "hey_jarvis"

# Machine-access capabilities (docs/architecture/machine-access.md). Knowing the
# names here lets configuration reject typos early; the tools that use them land
# in Month 1 Week 3 and Months 2-3.
KNOWN_CAPABILITIES = frozenset({"screen", "camera", "cursor", "keyboard", "clipboard", "apps"})


class Settings(BaseSettings):
    """Settings sourced from ``CLAIREAI_*`` environment variables or ``.env``.

    The default brain is Qwen3 1.7B served by a local Ollama daemon. That is a
    loopback HTTP call, not a cloud request: the model and the conversation stay
    on this machine. The old from-scratch checkpoint remains available as
    ``scratch`` for the research artifacts, but is no longer the default.
    """

    model_config = SettingsConfigDict(
        env_prefix="CLAIREAI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",
    )

    model_checkpoint: Path | None = None
    max_output_tokens: int = Field(default=128, ge=8, le=1024)
    owner_name: str = "Bikorimana Ange Tresor"

    # Brain provider. Ollama is the default because it runs the pretrained Qwen
    # models locally without requiring this project to train anything. `scratch`
    # is kept for the archived research checkpoints; `openai` and `anthropic`
    # remain opt-in hosted connectors.
    brain_provider: Literal["scratch", "ollama", "openai", "anthropic"] = "ollama"
    brain_model: str | None = None
    # Qwen2.5-Coder is used for code-heavy turns; `brain_model` handles the rest.
    brain_coding_model: str = DEFAULT_OLLAMA_CODING_MODEL
    brain_base_url: str | None = None
    brain_api_key: SecretStr | None = None
    # CPU prompt processing is the dominant cost on the target machine. A
    # file-path request once sent 25 schemas because routing missed the
    # extension, and a 39-token call took 112.8 s. Routing now narrows that
    # case, but 60 s was still too close to the real generation budget.
    brain_timeout_seconds: float = Field(default=180.0, gt=0, le=600)
    brain_max_tokens: int = Field(default=1024, ge=1, le=16384)
    # Qwen3 emits hidden reasoning by default. On this CPU that turned a 2-token
    # answer into 7.5 s instead of 1.8 s, and tool calls are already structured.
    # The owner can opt back in for deliberately difficult reasoning.
    brain_think: bool = False

    # Tool sandbox (roadmap 7.2). Empty by default: file tools can reach nothing
    # until the owner names the folders. Set as a JSON array, e.g.
    # CLAIREAI_ALLOWED_FOLDERS='["~/Downloads"]'.
    allowed_folders: list[Path] = Field(default_factory=list)
    agent_max_steps: int = Field(default=8, ge=1, le=24)

    # Tool routing (claireai.routing). `auto` narrows the schemas sent to the
    # model to the tools its intent can reach; `off` sends all of them. Routing
    # changes what the model is shown, never what it is allowed to do.
    tool_routing: Literal["auto", "off"] = "auto"

    # Presence renderer (docs/architecture/presence-renderer.md). `auto` means
    # WebGL when the webview supports it and holds the frame budget, else 2D.

    # Decoding temperature, chosen per phase by the agent loop rather than fixed:
    # while no tool has run the model is deciding what to do, which wants
    # repeatability; once it has real data it is phrasing an answer, which wants
    # natural language. Defaults are deliberately not 0 — measured, a smaller
    # model pinned at 0 emitted a malformed tool call on every single attempt.
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    temperature_tools: float = Field(default=0.2, ge=0.0, le=2.0)
    response_style: Literal["concise", "balanced", "detailed", "playful"] = "balanced"

    presence_mode: Literal["off", "auto", "2d", "webgl"] = "auto"
    presence_fps_cap: int = Field(default=30, ge=5, le=60)
    presence_reduce_motion: bool = False

    # Machine access is deny-by-default, like the folder allow-list.
    allowed_capabilities: list[str] = Field(default_factory=list)
    camera_device: int = Field(default=0, ge=0)

    # Local semantic memory. Ollama embeddings keep the text and vectors on this
    # machine; `hash` is an explicit offline fallback for constrained machines
    # and tests, not a silent substitute for the embedding model.
    memory_enabled: bool = True
    # Off by default: ordinary chat is already kept as conversation history.
    # When enabled, each completed turn is embedded locally as a `turn` entry so
    # semantic search can find it without a separate remember action.
    memory_auto_embed: bool = False
    embedding_provider: Literal["ollama", "hash"] = "ollama"
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_base_url: str | None = None
    embedding_timeout_seconds: float = Field(default=30.0, gt=0, le=300)

    # Wake-word detection is opt-in because it opens the microphone. The model
    # name is an installed openWakeWord model; no model is downloaded implicitly.
    wakeword_enabled: bool = False
    wakeword_model: str = DEFAULT_WAKEWORD_MODEL
    wakeword_threshold: float = Field(default=0.5, gt=0, le=1)
    # openWakeWord only knows its trained phrases (the default model is
    # "hey_jarvis"). "stt" uses the local faster-whisper model to hear an
    # arbitrary phrase such as "Hey Claire" in the owner's own voice.
    wakeword_engine: Literal["openwakeword", "stt"] = "stt"
    wakeword_phrase: str = "hey claire"

    # The HTTP API binds loopback unless the owner explicitly changes the host.
    # A bearer token is required for every route except /health.
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8765, ge=1, le=65535)
    api_token: SecretStr | None = None

    # Offline voice output is opt-in; pitch/rate map to the local eSpeak voice.
    voice_enabled: bool = False
    voice_pitch: int = Field(default=50, ge=0, le=99)
    voice_rate: int = Field(default=175, ge=80, le=450)

    # Development override. Production data belongs in the OS application-data
    # directory resolved by ``default_data_dir``.
    data_dir: Path | None = None
    log_level: str = "INFO"

    @model_validator(mode="after")
    def _validate_settings(self) -> "Settings":
        owner_name = self.owner_name.strip()
        if not owner_name:
            raise ValueError("CLAIREAI_OWNER_NAME must not be empty")
        object.__setattr__(self, "owner_name", owner_name)
        cleaned: list[Path] = []
        for folder in self.allowed_folders:
            text = str(folder).strip()
            if not text:
                raise ValueError("CLAIREAI_ALLOWED_FOLDERS must not contain empty entries")
            cleaned.append(Path(text).expanduser())
        object.__setattr__(self, "allowed_folders", cleaned)

        capabilities = [str(name).strip().lower() for name in self.allowed_capabilities]
        unknown = sorted({name for name in capabilities if name not in KNOWN_CAPABILITIES})
        if unknown:
            raise ValueError(
                f"CLAIREAI_ALLOWED_CAPABILITIES contains unknown capability {unknown[0]!r}. "
                f"Known capabilities: {', '.join(sorted(KNOWN_CAPABILITIES))}"
            )
        object.__setattr__(self, "allowed_capabilities", list(dict.fromkeys(capabilities)))

        provider = self.brain_provider
        if provider == "ollama" and not (self.brain_model or "").strip():
            object.__setattr__(self, "brain_model", DEFAULT_OLLAMA_MODEL)
        coding_model = (self.brain_coding_model or "").strip()
        if provider == "ollama" and not coding_model:
            coding_model = DEFAULT_OLLAMA_CODING_MODEL
        object.__setattr__(self, "brain_coding_model", coding_model)
        if provider != "scratch":
            model = (self.brain_model or "").strip()
            if not model:
                raise ValueError(
                    f"CLAIREAI_BRAIN_MODEL is required when CLAIREAI_BRAIN_PROVIDER={provider}"
                )
            object.__setattr__(self, "brain_model", model)
            endpoint = str(self.brain_base_url or DEFAULT_BASE_URLS[provider]).strip()
            object.__setattr__(self, "brain_base_url", check_endpoint(endpoint))
            if provider in {"openai", "anthropic"} and not self.brain_api_key:
                raise ValueError(
                    f"CLAIREAI_BRAIN_API_KEY is required when CLAIREAI_BRAIN_PROVIDER={provider}"
                )

        embedding_model = (self.embedding_model or "").strip()
        if not embedding_model:
            embedding_model = DEFAULT_EMBEDDING_MODEL
        object.__setattr__(self, "embedding_model", embedding_model)
        if self.embedding_provider == "ollama":
            endpoint = str(
                self.embedding_base_url
                or self.resolved_brain_base_url
                or DEFAULT_BASE_URLS["ollama"]
            ).strip()
            object.__setattr__(self, "embedding_base_url", check_endpoint(endpoint))

        wakeword_model = (self.wakeword_model or "").strip()
        if not wakeword_model:
            raise ValueError("CLAIREAI_WAKEWORD_MODEL must not be empty")
        object.__setattr__(self, "wakeword_model", wakeword_model)
        wakeword_phrase = (self.wakeword_phrase or "").strip()
        if not wakeword_phrase:
            raise ValueError("CLAIREAI_WAKEWORD_PHRASE must not be empty")
        object.__setattr__(self, "wakeword_phrase", wakeword_phrase)
        api_host = (self.api_host or "").strip()
        if not api_host:
            raise ValueError("CLAIREAI_API_HOST must not be empty")
        object.__setattr__(self, "api_host", api_host)
        return self

    @property
    def tool_temperature(self) -> float:
        """Temperature while the loop is still choosing a tool."""
        return self.temperature_tools

    @property
    def resolved_brain_base_url(self) -> str:
        """The connector endpoint, defaulted per provider and validated as https or loopback."""
        if self.brain_base_url:
            return self.brain_base_url
        return DEFAULT_BASE_URLS.get(self.brain_provider, "")

    @property
    def resolved_embedding_base_url(self) -> str:
        """Where local embeddings are requested; defaults to the Ollama endpoint."""
        return self.embedding_base_url or self.resolved_brain_base_url

    @property
    def api_token_value(self) -> str | None:
        """The bearer token itself, never serialised into logs."""
        return self.api_token.get_secret_value() if self.api_token else None

    @property
    def brain_api_key_value(self) -> str | None:
        """The key itself, for the HTTP adapter. Never logged or serialised."""
        if self.brain_api_key:
            return self.brain_api_key.get_secret_value()
        # Keyring before the environment variable: a key stored there is not
        # visible in `ps e`, a shell history or a dumped environment.
        from claireai.secrets import stored_secret

        return stored_secret(f"brain_api_key:{self.brain_provider}")

    @property
    def default_data_dir(self) -> Path:
        """OS-specific application-data directory (not the working directory)."""
        return Path(user_data_path(APP_NAME))

    @property
    def resolved_data_dir(self) -> Path:
        return Path(self.data_dir).expanduser() if self.data_dir else self.default_data_dir

    @property
    def database_path(self) -> Path:
        return self.resolved_data_dir / "db" / "claireai.sqlite3"

    @property
    def model_directory(self) -> Path:
        return self.resolved_data_dir / "models"

    @property
    def resolved_model_checkpoint(self) -> Path:
        """An explicit checkpoint, else the canonical one, else the newest run.

        A fresh install has no checkpoint, so the path still has to be named for
        the error to be useful. Once a data directory holds checkpoints from a
        training run, pointing the app at the newest complete one means an
        installed copy starts working without an environment variable; partial
        downloads and sidecar manifests are ignored.
        """
        if self.model_checkpoint:
            return Path(self.model_checkpoint).expanduser()
        canonical = self.model_directory / "claire-scratch.pt"
        if canonical.is_file():
            return canonical
        candidates = [
            path
            for path in self.model_directory.glob("claire-*.pt")
            if path.is_file() and not path.name.endswith(".partial")
        ]
        if candidates:
            return max(candidates, key=lambda path: path.stat().st_mtime)
        return canonical

    @property
    def control_state_path(self) -> Path:
        return self.resolved_data_dir / "runtime-control.json"

    @property
    def log_dir(self) -> Path:
        return self.resolved_data_dir / "logs"

    @property
    def allowed_folder_roots(self) -> tuple[Path, ...]:
        """Allow-list entries as absolute paths, without requiring them to exist."""
        return tuple(folder.expanduser().resolve(strict=False) for folder in self.allowed_folders)

    def ensure_directories(self) -> None:
        for directory in (self.database_path.parent, self.log_dir):
            directory.mkdir(parents=True, exist_ok=True)
            if os.name != "nt":
                try:
                    directory.chmod(0o700)
                except OSError:
                    pass
