"""Validated runtime configuration.

Configuration is validated once at startup and fails loudly: a partially
configured assistant that silently points at the wrong model or the wrong data
directory is worse than one that refuses to start.
"""

from __future__ import annotations

import ipaddress
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from platformdirs import user_data_path
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

APP_NAME = "ClaireAI"
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})

# Machine-access capabilities (docs/architecture/machine-access.md). Knowing the
# names here lets configuration reject typos early; the tools that use them land
# in Month 1 Week 3 and Months 2-3.
KNOWN_CAPABILITIES = frozenset({"screen", "camera", "cursor", "keyboard", "clipboard"})


def _is_loopback(host: str | None) -> bool:
    if not host:
        return False
    if host in LOOPBACK_HOSTS:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class Settings(BaseSettings):
    """Settings sourced from ``CLAIREAI_*`` environment variables or ``.env``.

    The model runtime defaults to loopback on purpose (roadmap 7.12). Pointing
    it at another machine requires an explicit opt-in so the default install
    never sends prompts over the network by accident.
    """

    model_config = SettingsConfigDict(
        env_prefix="CLAIREAI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",
    )

    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:1.5b"
    ollama_timeout_seconds: float = Field(default=120.0, ge=1.0, le=600.0)
    allow_non_loopback_ollama: bool = False

    # Tool sandbox (roadmap 7.2). Empty by default: file tools can reach nothing
    # until the owner names the folders. Set as a JSON array, e.g.
    # CLAIREAI_ALLOWED_FOLDERS='["~/Downloads"]'.
    allowed_folders: list[Path] = Field(default_factory=list)
    agent_max_steps: int = Field(default=8, ge=1, le=24)

    # Presence renderer (docs/architecture/presence-renderer.md). `auto` means
    # WebGL when the webview supports it and holds the frame budget, else 2D.
    # Cap on generated tokens per model call. Measured on this laptop: an
    # unbounded answer cost 126 s; a tool-using reply needs far less than 400.
    max_output_tokens: int = Field(default=400, ge=32, le=4096)

    presence_mode: Literal["off", "auto", "2d", "webgl"] = "auto"
    presence_fps_cap: int = Field(default=30, ge=5, le=60)
    presence_reduce_motion: bool = False

    # Machine access is deny-by-default, like the folder allow-list.
    allowed_capabilities: list[str] = Field(default_factory=list)
    camera_device: int = Field(default=0, ge=0)

    # Development override. Production data belongs in the OS application-data
    # directory resolved by ``default_data_dir``.
    data_dir: Path | None = None
    log_level: str = "INFO"

    @model_validator(mode="after")
    def _validate_runtime_endpoint(self) -> "Settings":
        url = self.ollama_base_url.strip().rstrip("/")
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError(
                f"CLAIREAI_OLLAMA_BASE_URL must be an http(s) URL with a host, got {self.ollama_base_url!r}"
            )
        if not _is_loopback(parsed.hostname) and not self.allow_non_loopback_ollama:
            raise ValueError(
                "CLAIREAI_OLLAMA_BASE_URL points at a non-loopback host "
                f"({parsed.hostname!r}). Local inference is the default; set "
                "CLAIREAI_ALLOW_NON_LOOPBACK_OLLAMA=true only after a security review."
            )
        if not self.ollama_model.strip():
            raise ValueError("CLAIREAI_OLLAMA_MODEL must not be empty")
        object.__setattr__(self, "ollama_base_url", url)

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
        return self

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
    def log_dir(self) -> Path:
        return self.resolved_data_dir / "logs"

    @property
    def allowed_folder_roots(self) -> tuple[Path, ...]:
        """Allow-list entries as absolute paths, without requiring them to exist."""
        return tuple(folder.expanduser().resolve(strict=False) for folder in self.allowed_folders)

    def ensure_directories(self) -> None:
        for directory in (self.database_path.parent, self.log_dir):
            directory.mkdir(parents=True, exist_ok=True)
